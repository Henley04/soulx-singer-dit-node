// binding.cc — Node.js N-API binding for SoulX-Singer DiT inference.
//
// Exposes a single JS class `Model` plus a top-level `getVersion()` function:
//
//   const { Model, getVersion } = require('./soulx_singer_dit.node');
//   getVersion();                                  // -> "0.1.0"
//   const m = new Model('/path/to/model.gguf', { backend: 'cpu' });
//   //   or: const m = Model.loadModel('/path/to/model.gguf', { backend: 'cpu' });
//
//   const v = m.forward({ x: Float32Array, cond: Float32Array, t: 0.5, T: 64 });
//   //   v: Float32Array of length MEL_DIM * T
//
//   const mel = m.reverseDiffusion({
//     promptMel:  Float32Array,  // [MEL_DIM, promptLen]
//     cond:       Float32Array,  // [HIDDEN, promptLen + targetLen]
//     z:          Float32Array,  // [MEL_DIM, targetLen]
//     promptLen:  16,
//     targetLen:  48,
//     nSteps:     8,
//     seed:       12345,         // optional; defaults to 12345
//   });
//   //   mel: Float32Array of length MEL_DIM * targetLen
//
//   m.release();   // free native resources eagerly (optional; GC also works)
//
// The `backend` option is informational only at the C++ layer — the JS layer
// is responsible for loading the right .node file per backend. The C++ binding
// always loads the GGUF via mmap and runs inference through ggml_mul_mat on
// quantized weights (no dequantization).

#include <napi.h>

#include "infer.h"

#include <cstring>
#include <string>
#include <vector>

namespace {

// -------------------------------------------------------------------------
// Small argument-extraction helpers. Each returns false (after throwing a
// TypeError) on failure, so callers can early-return.
// -------------------------------------------------------------------------

bool GetFloat32Array(Napi::Env env,
                     const Napi::Object& opts,
                     const char* key,
                     Napi::Float32Array& out) {
    if (!opts.Has(key)) {
        Napi::TypeError::New(env, std::string("missing field: ") + key)
            .ThrowAsJavaScriptException();
        return false;
    }
    Napi::Value v = opts.Get(key);
    if (!v.IsTypedArray()) {
        Napi::TypeError::New(env, std::string(key) + " must be a Float32Array")
            .ThrowAsJavaScriptException();
        return false;
    }
    Napi::TypedArray ta = v.As<Napi::TypedArray>();
    if (ta.TypedArrayType() != napi_float32_array) {
        Napi::TypeError::New(env, std::string(key) + " must be a Float32Array")
            .ThrowAsJavaScriptException();
        return false;
    }
    out = v.As<Napi::Float32Array>();
    return true;
}

bool GetInt32(Napi::Env env,
              const Napi::Object& opts,
              const char* key,
              int* out) {
    if (!opts.Has(key)) {
        Napi::TypeError::New(env, std::string("missing field: ") + key)
            .ThrowAsJavaScriptException();
        return false;
    }
    Napi::Value v = opts.Get(key);
    if (!v.IsNumber()) {
        Napi::TypeError::New(env, std::string(key) + " must be a number")
            .ThrowAsJavaScriptException();
        return false;
    }
    *out = v.As<Napi::Number>().Int32Value();
    return true;
}

bool GetFloat(Napi::Env env,
              const Napi::Object& opts,
              const char* key,
              float* out) {
    if (!opts.Has(key)) {
        Napi::TypeError::New(env, std::string("missing field: ") + key)
            .ThrowAsJavaScriptException();
        return false;
    }
    Napi::Value v = opts.Get(key);
    if (!v.IsNumber()) {
        Napi::TypeError::New(env, std::string(key) + " must be a number")
            .ThrowAsJavaScriptException();
        return false;
    }
    *out = v.As<Napi::Number>().FloatValue();
    return true;
}

}  // namespace

// -------------------------------------------------------------------------
// ModelWrapper — wraps a C++ Model as a JS class via Napi::ObjectWrap.
// -------------------------------------------------------------------------
class ModelWrapper : public Napi::ObjectWrap<ModelWrapper> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports) {
        Napi::Function func = DefineClass(env, "Model", {
            InstanceMethod("forward",          &ModelWrapper::Forward),
            InstanceMethod("reverseDiffusion", &ModelWrapper::ReverseDiffusion),
            InstanceMethod("release",          &ModelWrapper::Release),
            StaticMethod("loadModel",          &ModelWrapper::LoadModel),
        });

        // Persist the constructor so the static `loadModel` factory can build
        // new instances. Stored as env instance data with a finalizer so it is
        // freed when the env tears down.
        auto* ctor = new Napi::FunctionReference();
        *ctor = Napi::Persistent(func);
        env.SetInstanceData<Napi::FunctionReference>(ctor);

        exports.Set(Napi::String::New(env, "Model"), func);
        exports.Set(Napi::String::New(env, "getVersion"),
                    Napi::Function::New(env, &ModelWrapper::GetVersion));
        return exports;
    }

    // Constructor: new Model(path: string, options?: { backend?: string })
    // The `backend` option is accepted but ignored at the C++ layer (it is
    // used by the JS layer to pick the correct .node file).
    ModelWrapper(const Napi::CallbackInfo& info)
        : Napi::ObjectWrap<ModelWrapper>(info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsString()) {
            Napi::TypeError::New(env,
                "Model constructor requires (path: string, options?: object)")
                .ThrowAsJavaScriptException();
            return;
        }
        std::string path = info[0].As<Napi::String>().Utf8Value();

        // Optional options object — validate but otherwise ignored.
        if (info.Length() >= 2 && !info[1].IsUndefined() && !info[1].IsNull()) {
            if (!info[1].IsObject()) {
                Napi::TypeError::New(env, "options must be an object")
                    .ThrowAsJavaScriptException();
                return;
            }
            // `backend` field (if present) is informational only.
        }

        if (!model_.load(path)) {
            Napi::Error::New(env, "Failed to load GGUF model: " + path)
                .ThrowAsJavaScriptException();
            return;
        }
    }

    ~ModelWrapper() {
        // Deterministic cleanup if release() was not called explicitly.
        model_.unload();
    }

private:
    Model model_;

    // ---------------------------------------------------------------------
    // Static factory: Model.loadModel(path, options?) -> Model
    // Delegates to the constructor.
    // ---------------------------------------------------------------------
    static Napi::Value LoadModel(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        auto* ctor = env.GetInstanceData<Napi::FunctionReference>();
        if (!ctor) {
            Napi::Error::New(env, "Model class not initialized")
                .ThrowAsJavaScriptException();
            return env.Null();
        }
        std::vector<napi_value> args;
        args.reserve(info.Length());
        for (size_t i = 0; i < info.Length(); i++) args.push_back(info[i]);
        return ctor->New(args);
    }

    // ---------------------------------------------------------------------
    // getVersion() -> "0.1.0"
    // ---------------------------------------------------------------------
    static Napi::Value GetVersion(const Napi::CallbackInfo& info) {
        return Napi::String::New(info.Env(), "0.1.0");
    }

    // ---------------------------------------------------------------------
    // model.forward({ x, cond, t, T }) -> Float32Array (MEL_DIM * T)
    // ---------------------------------------------------------------------
    Napi::Value Forward(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (model_.ctx == nullptr) {
            Napi::Error::New(env, "Model is not loaded").ThrowAsJavaScriptException();
            return env.Null();
        }
        if (info.Length() < 1 || !info[0].IsObject()) {
            Napi::TypeError::New(env, "forward requires an options object")
                .ThrowAsJavaScriptException();
            return env.Null();
        }
        Napi::Object opts = info[0].As<Napi::Object>();

        Napi::Float32Array x;
        Napi::Float32Array cond;
        float t = 0.0f;
        int   T = 0;
        if (!GetFloat32Array(env, opts, "x",    x))    return env.Null();
        if (!GetFloat32Array(env, opts, "cond", cond)) return env.Null();
        if (!GetFloat(env, opts, "t", &t))             return env.Null();
        if (!GetInt32(env, opts, "T", &T))             return env.Null();

        if (T <= 0) {
            Napi::TypeError::New(env, "forward: T must be positive")
                .ThrowAsJavaScriptException();
            return env.Null();
        }
        if (x.ElementLength() != (size_t)MEL_DIM * (size_t)T) {
            Napi::TypeError::New(env,
                "forward: x.length must equal MEL_DIM * T (128 * T)")
                .ThrowAsJavaScriptException();
            return env.Null();
        }
        if (cond.ElementLength() != (size_t)HIDDEN * (size_t)T) {
            Napi::TypeError::New(env,
                "forward: cond.length must equal HIDDEN * T (1024 * T)")
                .ThrowAsJavaScriptException();
            return env.Null();
        }

        std::vector<float> out = run_forward(model_, x.Data(), cond.Data(), t, T);

        Napi::Float32Array result = Napi::Float32Array::New(env, out.size());
        std::memcpy(result.Data(), out.data(), out.size() * sizeof(float));
        return result;
    }

    // ---------------------------------------------------------------------
    // model.reverseDiffusion({
    //   promptMel, cond, z, promptLen, targetLen, nSteps, seed?
    // }) -> Float32Array (MEL_DIM * targetLen)
    // ---------------------------------------------------------------------
    Napi::Value ReverseDiffusion(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (model_.ctx == nullptr) {
            Napi::Error::New(env, "Model is not loaded").ThrowAsJavaScriptException();
            return env.Null();
        }
        if (info.Length() < 1 || !info[0].IsObject()) {
            Napi::TypeError::New(env, "reverseDiffusion requires an options object")
                .ThrowAsJavaScriptException();
            return env.Null();
        }
        Napi::Object opts = info[0].As<Napi::Object>();

        Napi::Float32Array promptMel;
        Napi::Float32Array cond;
        Napi::Float32Array z;
        int promptLen = 0;
        int targetLen = 0;
        int nSteps    = 0;
        if (!GetFloat32Array(env, opts, "promptMel", promptMel)) return env.Null();
        if (!GetFloat32Array(env, opts, "cond",      cond))      return env.Null();
        if (!GetFloat32Array(env, opts, "z",         z))         return env.Null();
        if (!GetInt32(env, opts, "promptLen", &promptLen))       return env.Null();
        if (!GetInt32(env, opts, "targetLen", &targetLen))       return env.Null();
        if (!GetInt32(env, opts, "nSteps",    &nSteps))          return env.Null();

        if (promptLen <= 0 || targetLen <= 0 || nSteps <= 0) {
            Napi::TypeError::New(env,
                "reverseDiffusion: promptLen, targetLen and nSteps must be positive")
                .ThrowAsJavaScriptException();
            return env.Null();
        }
        if (promptMel.ElementLength() != (size_t)MEL_DIM * (size_t)promptLen) {
            Napi::TypeError::New(env,
                "reverseDiffusion: promptMel.length must equal MEL_DIM * promptLen")
                .ThrowAsJavaScriptException();
            return env.Null();
        }
        if (z.ElementLength() != (size_t)MEL_DIM * (size_t)targetLen) {
            Napi::TypeError::New(env,
                "reverseDiffusion: z.length must equal MEL_DIM * targetLen")
                .ThrowAsJavaScriptException();
            return env.Null();
        }
        int T = promptLen + targetLen;
        if (cond.ElementLength() != (size_t)HIDDEN * (size_t)T) {
            Napi::TypeError::New(env,
                "reverseDiffusion: cond.length must equal HIDDEN * (promptLen + targetLen)")
                .ThrowAsJavaScriptException();
            return env.Null();
        }

        // Optional seed (default 12345 keeps parity with infer_orig.cpp).
        uint64_t seed = 12345ULL;
        if (opts.Has("seed") && opts.Get("seed").IsNumber()) {
            seed = (uint64_t)opts.Get("seed").As<Napi::Number>().Int64Value();
        }

        std::vector<float> promptMelVec(promptMel.Data(),
                                        promptMel.Data() + promptMel.ElementLength());
        std::vector<float> condVec(cond.Data(),
                                   cond.Data() + cond.ElementLength());
        std::vector<float> zVec(z.Data(),
                                z.Data() + z.ElementLength());

        RNG rng(seed);
        std::vector<float> out = run_reverse_diffusion(
            model_, promptMelVec, condVec, zVec,
            promptLen, targetLen, nSteps, rng);

        Napi::Float32Array result = Napi::Float32Array::New(env, out.size());
        std::memcpy(result.Data(), out.data(), out.size() * sizeof(float));
        return result;
    }

    // ---------------------------------------------------------------------
    // model.release() — free native resources eagerly.
    // ---------------------------------------------------------------------
    Napi::Value Release(const Napi::CallbackInfo& info) {
        model_.unload();
        return info.Env().Undefined();
    }
};

// -------------------------------------------------------------------------
// Module registration
// -------------------------------------------------------------------------
Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
    return ModelWrapper::Init(env, exports);
}

NODE_API_MODULE(soulx_singer_dit, InitAll)
