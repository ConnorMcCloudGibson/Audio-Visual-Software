//#define CM7
// -Ofast -Wall -march=armv7-m -mcpu=cortex-m7 -fno-rtti -fno-exceptions -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -std=c++17 -fsingle-precision-constant -DCM7

#define X86
// -Ofast -march=skylake -fno-rtti -fno-exceptions -fsingle-precision-constant -std=c++17 -DX86
// i7-8565u has SSE4.1, SSE4.2, AVX2


#pragma once
#include <stdint.h>
#include <cmath>
// TODO replace these to decrease dependencies and speed up compilation? mostly math functions

// TODO look for where int/fixed point implementations would be favourable
// TODO utils/primitives for seq/tempo related stuff

////////////////////////////////////////////////////////// UTILS

///////////////////////////// CLIP / SCALE

// TODO add arbitrary clamp, scale, min/max
// int min/max.. hmm no arm instructions.. x86 has packed int min/max but value needs to be in SSE/AVX regs

inline float clip_u(float x){ // clip to [0,1]
    if(x>1.0f){x=1.0f;}
    if(x<0.0f){x=0.0f;}
    return x;
}

inline float clip_b(float x){ // clip to [-1,1]
    if(x>1.0f){x=1.0f;}
    if(x<-1.0f){x=-1.0f;}
    return x;
}

inline float halfwave_pos(float x){ // positive halfwave rectification
    return (x>0.0f) ? x : 0.0f;
}

inline float halfwave_neg(float x){ // negative halfwave rectification
    return (x<0.0f) ? x : 0.0f;
}

inline float halfwave_posinv(float x){ // halfwave - positive inverted to negative
    x = -x;
    return (x<0.0f) ? x : 0.0f;
}

inline float halfwave_neginv(float x){ // halfwave - negative inverted to positive
    x = -x;
    return (x>0.0f) ? x : 0.0f;
}

inline float pscale_left(float x){
    float calc = -2.0f * x + 1.0f;
    return (calc > 0.0f) ? calc : 0.0f;
}

inline float pscale_right(float x){
    float calc = 2.0f * x - 1.0f;
    return (calc > 0.0f) ? calc : 0.0f;
}

inline float pscale_mid(float x){
    return 2.0f*(0.5f - fabsf(x - 0.5f));
}

inline float u2bi(float x){return 2.0f*x - 1.0f;}
inline float bi2u(float x){return 0.5f*x + 0.5f;}
inline float atnv(float x){return x*fabsf(x);}

inline float unit(float x){return x;}
inline float square(float x){return x*x;}
inline float cube(float x){return x*x*x;}
inline float quad(float x){return x*x*x*x;}
inline float quint(float x){return x*x*x*x*x;}

inline float isquare(float x){return x*(2.0f - x);} // equivalent to 1 - (1 - x)^2
// TODO add 'inverted' versions of other funcs - i.e. 1 - f(1-x)

inline float aclip(float x){ // abs softclip - quite sharp knee
    return x / (1.0f + fabsf(x));
}

inline float fclip(float x){ // simple folding clip
    return 2.0f*x / (1.0f + x*x);
}

inline float polyclip(float x){ // (-1,-1) and (1,1) with zero derivative at each
    return 0.5f*(3.0f-x*x)*x;
}

// TODO FMODF rounding behaviour?
inline float polysin_bi(float x){ // sin approx bipolar input
    return 2.5f * x * (1.0f - x * x);
}

inline float polysin_uni(float x){ // sin approx unipolar input
    //return 20.0f * x * (x - 0.5f)*(x - 1.0f);
    return x * (x * (x * 20.0f - 30.0f) + 10.0f);
}

///////////////////////////// ROUNDING

enum Roundtype {rCURR, rZERO, rNEAR, rINFP, rINFN};

template<Roundtype R = rINFN>
inline float round_f2f(float x){
#ifdef CM7

    if(R == rCURR){
        asm("vrintr.f32 %0, %1" : "=t"(x) : "t"(x));
    } else if(R == rZERO){
        asm("vrintz.f32 %0, %1" : "=t"(x) : "t"(x));
    } else if(R == rNEAR){
        asm("vrinta.f32 %0, %1" : "=t"(x) : "t"(x));
    } else if(R == rINFP){
        asm("vrintp.f32 %0, %1" : "=t"(x) : "t"(x));
    } else if(R == rINFN){
        asm("vrintm.f32 %0, %1" : "=t"(x) : "t"(x));
    }
    return x;

#else
#ifdef X86

    // _MM_FROUND_NO_EXC == 8 - suppresses fp exceptions for this instr
    if(R == rCURR){ //_MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC
        asm("vroundss %0, %0, %0, 12" : "=x"(x) : "x"(x));
    } else if(R == rZERO){ //_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC
        asm("vroundss %0, %0, %0, 11" : "=x"(x) : "x"(x));
    } else if(R == rNEAR){ //_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
        asm("vroundss %0, %0, %0, 8 " : "=x"(x) : "x"(x));
    } else if(R == rINFP){ //_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC
        asm("vroundss %0, %0, %0, 10" : "=x"(x) : "x"(x));
    } else if(R == rINFN){ //_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC
        asm("vroundss %0, %0, %0, 9 " : "=x"(x) : "x"(x));
    }
    return x;

#else

    if(R == rCURR){
        return nearbyintf(x);
    } else if(R == rZERO){
        return truncf(x);
    } else if(R == rNEAR){
        //return roundf(x); // vroundss instr not emitted for gcc x86
        // this sometimes optimises out though:
        return ceilf(x + 0.5f);
    } else if(R == rINFP){
        return ceilf(x);
    } else if(R == rINFN){
        return floorf(x);
    }

#endif
#endif
}


template<Roundtype R = rINFN>
inline int32_t round_f2i(float x){
#ifdef CM7

	int32_t out;
    if(R == rCURR){
        asm("vcvtr.s32.f32  %0, %1" : "=t"(x) : "t"(x) : );
        asm("vmov  %0, %1" : "=r"(out) : "t"(x) : );
    } else if(R == rZERO){
        asm("vcvt.s32.f32  %0, %1" : "=t"(x) : "t"(x) : );
        asm("vmov  %0, %1" : "=r"(out) : "t"(x) : );
    } else if(R == rNEAR){
        asm("vcvta.s32.f32  %0, %1" : "=t"(x) : "t"(x) : );
        asm("vmov  %0, %1" : "=r"(out) : "t"(x) : );
    } else if(R == rINFP){
        asm("vcvtp.s32.f32  %0, %1" : "=t"(x) : "t"(x) : );
        asm("vmov  %0, %1" : "=r"(out) : "t"(x) : );
    } else if(R == rINFN){
        asm("vcvtm.s32.f32  %0, %1" : "=t"(x) : "t"(x) : );
        asm("vmov  %0, %1" : "=r"(out) : "t"(x) : );
    }
	return out;

#else
#ifdef X86

    int32_t out;
    if(R == rCURR){
        asm("vcvtss2si %0, %1" : "=r"(out) : "x"(x) : );
    } else if(R == rZERO){
        asm("vcvttss2si %0, %1" : "=r"(out) : "x"(x) : );
    } else if(R == rNEAR){
        asm("vcvtss2si %0, %1, %{rn-sae}" : "=r"(out) : "x"(x) : );
    } else if(R == rINFP){
        asm("vcvtss2si %0, %1, %{ru-sae}" : "=r"(out) : "x"(x) : );
    } else if(R == rINFN){
        asm("vcvtss2si %0, %1, %{rd-sae}" : "=r"(out) : "x"(x) : );
    }
    return out;

#else

    return round_f2f<R>(x);

#endif
#endif
}


inline float fmod1(float x){ // float modulo 1.0f
    return x - round_f2f<rINFN>(x);
}
float fmod1_bi(float x){ // float modulo - bipolar wraparound (-1,1)
    return x - 2.0f * round_f2f<rNEAR>(0.5f * x);
}


///////////////////////////// SIGNAL CONVERSIONS

template<typename Td = float, typename Ts>
inline Td signal_cast(Ts src);

template<> inline float signal_cast<float>(float x){
    return x;
}

template<> inline float signal_cast<float>(int16_t x){
    return (float)(x/32768.0f);
}

template<> inline float signal_cast<float>(int8_t x){
    return (float)(x/128.0f);
}

template<> inline int16_t signal_cast<int16_t>(float x){
    return round_f2i(32767.5f * x);
}

// TODO fill the rest in
// TODO some of the x86 assembly is weird here
// 32bit useful for templated RNG
// extrema.. ignore the negative-most? and have all multipliers as 2^n-1 values?
// rounding? for f2i
// companding deeper dive.. look into quantised versions. on wikipedia


////////////////////////////////////////////////////////// INTERPOLATION

// linear and others derived from it: f(1-u)*x + f(u)*y, 
// where f : [0,1] -> [0,1], monotonic increasing (though could be interesting if it isn't)

inline float interp_lin(float x, float y, float u){
    return x + u * (y - x);
}

// TODO check performance of doing interpolation as integers then converting to float
// as is these are redundant - could just cast and do normal linear interp

inline float interp_lin(int8_t x, int8_t y, float u){
    float xf = x/128.0f;
    float yf = y/128.0f;
    return xf + (yf - xf) * u;
}

inline float interp_lin(int16_t x, int16_t y, float u){
    float xf = x/32768.0f;
    float yf = y/32768.0f;
    return xf + (yf - xf) * u;
}

inline float sstp(float x){ // 'smoothstep'
    return x * x * (3.0f - 2.0f * x);
}

// note 1-sstp(x) = sstp(1-x)
inline float interp_ss(float x, float y, float u){
    return x + (y - x) * sstp(u);
}

inline float interp_ep(float x, float y, float u){
    return sqrtf(1.0f-u)*x + sqrtf(u)*y;
}
// TODO is this broken? i had some issues with it on arm.. be wary of negative u

inline float interp_sq(float x, float y, float u){
	//f(x) = 1-x^2
    float u2 = u*u;
    return (1.0f - u2)*x + (2.0f*u - u2)*y;
}

// multi-arg.. interpolation window between y1 and y2
// ABI limits?? how many float args?

inline float interp_cubic(float y0, float y1, float y2, float y3, float u){
	float u2 = u*u;
	float a0 = y3 - y2 - y0 + y1;
	float a1 = y0 - y1 - a0;
	float a2 = y2 - y0;
	float a3 = y1;
	return a0*u*u2 + a1*u2 + a2*u + a3;
    //return y1 + 0.5f * u*(y2 - y0 + u*(2.0f*y0 - 5.0f*y1 + 4.0f*y2 - y3 + u*(3.0f*(y1 - y2) + y3 - y0)));
}
// TODO check validity.....

inline float interp_crom(float y0, float y1, float y2, float y3, float u){ //catmull-rom
	float u2 = u*u;
	float a0 = -0.5*y0 + 1.5*y1 - 1.5*y2 + 0.5*y3;
	float a1 = y0 - 2.5*y1 + 2*y2 - 0.5*y3;
	float a2 = -0.5*y0 + 0.5*y2;
	float a3 = y1;
	return a0*u*u2 + a1*u2 + a2*u + a3;
}



////////////////////////////////////////////////////////// LUT
// generalised time domain transfer function class? separate out now with auto&
// lut, polynomial, more params.. wavetable, dynamic polynomials

// i can do this with auto& template parameter, don't need to change things here
// need to ensure static object lifetime

// TODO rewrite with int16/8 etc overloads of interp functions rather than casting then using float versions
// at least for linear

// TODO clean up variadic bit.. seems overly complicated.. narrowing conversion... 
// stupid constructor type like.. Lut<float, float, float, float, int, float, ...>

enum Interptype {NONE, LINEAR, CLINEAR, SS, CUBIC, CROM};

template<typename T, size_t n, Interptype t>
class Lut{
private:
    T a[n];
public:
    template<typename... Tval> constexpr Lut(Tval... values) : a{values...} {
        static_assert(sizeof...(values) == n, "number of lut values must match length of array");
    }

    inline float operator[](uint32_t x) const {
        return a[x % n];
    }

    inline float operator()(float U) const {
        if (t == NONE){
            int32_t Uq = round_f2i(U * (n-1));
            return signal_cast<float>(a[Uq % n]);

        } else if (t == LINEAR){
            int32_t Uq = round_f2i(U * n);
            float u = U * n - Uq;
            float x = signal_cast(a[Uq % n]);
            float y = signal_cast(a[(Uq+1) % n]);
            return interp_lin(x, y, u);

        } else if (t == CLINEAR){ // U must be clipped 0<=U<=1 !! no wraparound
            uint32_t Uq = (uint32_t)(U * (n-1));
            float u = U * (n-1) - Uq;
            return interp_lin(signal_cast(a[Uq]), signal_cast(a[((Uq+1)>=n) ? (Uq) : (Uq+1)]), u);
            //TODO optimise this... there should be a vmax instruction in there
            
        } else if (t == SS){
            int32_t Uq = round_f2i(U * n);
            float u = U * n - Uq;
            return interp_ss(signal_cast(a[Uq % n]), signal_cast(a[(Uq+1) % n]), u);
            
        } else if (t == CUBIC) {
            int32_t Uq = round_f2i(U * n);
            float u = U * n - (float)Uq;
            float y0 = signal_cast(a[(Uq-1)&(n-1)]);
            float y1 = signal_cast(a[    Uq&(n-1)]);
            float y2 = signal_cast(a[(Uq+1)&(n-1)]);
            float y3 = signal_cast(a[(Uq+2)&(n-1)]);
            return interp_cubic(y0, y1, y2, y3, u);
    
        } else if (t == CROM) {
            int32_t Uq = round_f2i(U * n);
            float u = U * n - (float)Uq;
            float y0 = signal_cast(a[(Uq-1)&(n-1)]);
            float y1 = signal_cast(a[    Uq&(n-1)]);
            float y2 = signal_cast(a[(Uq+1)&(n-1)]);
            float y3 = signal_cast(a[(Uq+2)&(n-1)]);
            return interp_crom(y0, y1, y2, y3, u);
        } else {return U;}
    }
};

// lut<float, 256> lutA{
//     #include "voct_48k.h"
// };

// another way, same asm but above is better
// interesting method though, might be able to use this pattern elsewhere

// template<auto& a>
// struct lut{
//     inline float operator[](uint32_t x){
//         uint32_t n = (sizeof(a)/sizeof(a[0]));
//         return a[x&(n-1)];
//     }
// };

// float A[] = {
//     #include "voct_48k.h"
// };
// lut<A> lutA;




////////////////////////////////////////////////////////// FILTERS
// simple 1poles
// tpt 1poles
// FIR, averaging, tomisawa, differencing
// naive SVF
// tpt SVF
// ? implementations of - 4pole, 'phaser', 
// pole-mixing utilities, container for multi-filters?, routing/sum-difference/mixing
// ? hilbert/dome.. complex number shit

// TODO check performance of multi-arg functions.. input, frequency
//     note that store-forwarding could help a lot here.. in the case of directly consecutive write->read
// TODO check performance of const cutoff situation.. 
//     could make another simple version for smoothing/eq filters.. eh would like float nttp but whatever..
//     float immediate values for cutoffs?
// TODO bunch of these 1-poles seemingly have redundant move instructions in them hmm..
//    yeah just need to go thru these tight loops again
// TODO make consistent the initial value behaviour.. blah = 0.0f; or do it in constructor?


///////////////////////////// 1-POLE

template<auto& s = unit> // function or functor object for internal cutoff scaling
class Lp1 {
private:
    float state;
    float c;
public:
    Lp1(float c_ext) : state{0.0f}, c(s(c_ext)) {}
    Lp1(){}
    inline void operator[](float c_ext){
        c = s(c_ext);
    }
    inline float operator()(float in){ // TODO arm assembly looks a bit suboptimal, weird extra vmov
        return state += c * (in - state);
    }
    inline void reset(){state = 0.0f;}
};


template<auto& s = unit> // function or functor object for internal cutoff scaling
class Hp1 {
private:
    float state;
    float c;
public:
    Hp1(float c_ext) : state{0.0f}, c(s(c_ext)) {}
    Hp1(){}
    inline void operator[](float c_ext){
        c = s(c_ext);
    }
    inline float operator()(float in){ 
        state += c * (in - state);
        return in - state;
    }
    inline void reset(){state = 0.0f;}
};


template<auto& s = unit> // function or functor object for internal scaling
class Ap1{ //from two combs
private:
    float state;
    float c;
public:
    Ap1(float c_ext) : state{0.0f}, c(s(c_ext)){}
    Ap1(){}
    inline void operator[](float c_ext){
        c = s(c_ext);
    }
    inline float operator()(float in){
        float out = state - c*in;
            state = in    + c*out;
        return out;
    }
    inline void reset(){state = 0.0f;}
};


enum Filtype1 {LP1, HP1, AP1};

template<Filtype1 FTYPE, auto& scale = unit>
class Tpt1 {
private:
    float s = 0.0f;
    float g = 0.0f;
public:
    inline void operator[](float nf){
        g = scale(nf);
    }
    inline void set_g(float g){g = g;}
    inline float operator()(float x){
        if (FTYPE == LP1){
        float v = g * (x - s);
        float y = v + s;
              s = y + v;
        return y;

        } else if (FTYPE == HP1){
        float v = g * (x - s);
        float y = v + s;
              s = y + v;
        return x - y;

        } else if (FTYPE == AP1){
        float v = x - s;
        s += v * 2.0f * g;
        return s - v;
        }

    }
    inline void reset(){s = g = 0.0f;}
};


///////////////////////////// 2-POLE

enum Filtype2 {LP2, BP2, BPN, HP2, AP2, NO2, PEQ, LSH, HSH};

/* chamberlin/naive:
g = 2sin(pi*f/fs)
or use small angle g ~ 2*pi*f/fs
limit to g<1 (i.e. nf<1/6 or a bit less for the approximation)

R = 1/r // 'damping'
r: 0.5f to inf.. wait maybe factor of 2... so 1.0 to inf? it's blowing up near 0.5...
*/

// TODO work out resonance scaling

template<Filtype2 FTYPE, auto& scale = unit>
class Svfn{
private:
    float lp = 0.0f;
    float bp = 0.0f; // state variables
    float g, R;
    float gain = 1.0f;

public:
    inline void set_fc(float nf){g = scale(nf);} // 0<nf<0.159f
    inline void set_f_raw(float f){g = f;} // g ~= 2pi*f/SR
    inline void set_gain(float gain_ext){gain = gain_ext;}
    inline void set_r(float r_ext){R = 2.0f * (1.0f - r_ext);} // 0<r<1
    inline void set_r_raw(float r_ext){R = r_ext;} // 0<r<2???.. scaling??
    inline float operator()(float in){ 

        float hp, notch, bp_norm;
        bp_norm = bp * R;
        notch = in - bp_norm;
        lp += g * bp;
        hp = notch - lp;
        bp += g * hp;

        if(FTYPE == LP2){return lp;}
        else if(FTYPE == BP2){return bp;}
        else if(FTYPE == BPN){return bp_norm;}
        else if(FTYPE == HP2){return hp;}
        else if(FTYPE == AP2){return in - 2.0f * bp_norm;}
        else if(FTYPE == NO2){return notch;}
        else if(FTYPE == PEQ){return in + gain * bp;}
        else if(FTYPE == LSH){return in + gain * lp;}
        else if(FTYPE == HSH){return in + gain * hp;}

    }
    inline void reset(){lp = bp = 0.0f;}
};


template<Filtype2 FTYPE, auto& scale = unit>
class Svfz {
private:
    float s1 = 0.0f; 
    float s2 = 0.0f;
    float g, R;
    float gain;
public:
    inline void fr(float nf, float r){
        g = scale(nf); // g = tan(pi*f/fs)
        R = 1.0f/r; // factor of 2??? 1/2*resonance????????
        // r = 2.0f gives cancellation with gain = -1.. hmmm gain needs to be scaled to dB
    }
    inline float operator()(float in){
        float g1 = 2.0f*R + g;
        float d = 1.0f / (1.0f + 2.0f * R * g + g * g);
        float hp, bp, lp;
        hp = (in - g1 * s1 - s2) * d;
        float v1 = g * hp;
        bp = v1 + s1;
        s1 = bp + v1; // first integrator
        float v2 = g * bp;
        lp = v2 + s2;
        s2 = lp + v2; // second integrator

        if(FTYPE == LP2){return lp;}
        else if(FTYPE == BP2){return bp;}
        else if(FTYPE == BPN){return bp * R;}
        else if(FTYPE == HP2){return hp;}
        else if(FTYPE == AP2){return in - 2.0f * bp * R;}
        else if(FTYPE == NO2){return in - bp * R;}
        else if(FTYPE == PEQ){return in + gain * bp;}
        else if(FTYPE == LSH){return in + gain * lp;}
        else if(FTYPE == HSH){return in + gain * hp;}
    }
    inline void set_gain(float gain_ext){gain = gain_ext;}
    inline void reset(){s1 = s2 = 0.0f;}
};



////////////////////////////////////////////////////////// generators
// phasorb
// noise - lcg, xorshift
// blepz
// fixed point fast implementations?

template<auto& s = unit> // function or functor object for internal scaling
class Phasor{
private:
    float phase = 0.0f;
public:
    inline float operator()(float inc) {
        inc = s(inc);
        phase += inc;
        phase -= round_f2f(phase);
        return phase;
    }
};


inline float poly_blep(float t, float dt) {
    if (t < dt) {
        return -square(t / dt - 1.0f);

    } else if (t > 1.0f - dt) {
        return square((t - 1.0f) / dt + 1.0f);

    } else {
        return 0.0f;
    }
}


// these are the same.. does one perform better than the other??

inline float poly_blep2(float t, float dt) {
    if (t < dt) {
        t /= dt;
        return t+t - t*t - 1.;

    } else if (t > 1. - dt) {
        t = (t - 1.) / dt;
        return t*t + t+t + 1.;

    } else {
        return 0.;
    }
}

//template<lut<float> & scale>
struct pbsaw{

	float p = 0.0f;
	float pinc = 0.1f;

	inline float operator()(float nf){
		//pinc = scale(nf);
        pinc = nf;
		p += pinc;
		if (p>=1.0f){p -= 1.0f;}

		float t_ = p + 0.5f;
		if (t_ >= 1.0f) {t_ -= 1.0f;}

		float y = 2.0f * t_ - 1.0f;
		return y - poly_blep(t_, pinc);

	}
};

// can definitely optimise this a lot
// use same pinc etc. just use different wraparound

//template<lut<float> & scale>
struct pbsquare{

	float p = 0.0f;
	float pinc = 0.1f;

	inline float operator()(float nf, float pwin){

		//pinc = scale(nf);
        pinc = nf;
		p += pinc;
		if (p>=1.0f){p -= 1.0f;}

		float pw = 0.5f * (1.0f - pwin);  // BANDAID

		float t_ = p + 1.0f - pw;
		if (t_ >= 1.0f) {t_ -= 1.0f;}

		float y = -2.0f * pw;
		if (p < pw) {y += 2.0f;}
		return y + poly_blep(p, pinc) - poly_blep(t_, pinc);
	}

};



struct lcg{
    volatile int32_t state;
    int32_t a_, c_;
    lcg(int32_t seed, int32_t a = 16807, int32_t c = 0){
        state = seed;
        a_ = a;
        c_ = c;
    }
    inline float operator()(){
        state = a_*state + c_;
        return (float)(state * 4.6566129e-010f);
    }

};
/* other possible vals

catfact 'badrand'
a = 0x3c6ef35f
c = 0x19660d

monome dsp-kit lcg
a = 1597334677
c = 1289706101

pichenettes stmlib random.h
a = 1664525
c = 1013904223

*/




////////////////////////////////////////////////////////// delay line
// dline, allpass delay
// comb/waveguide abstraction
// helpers for chains/reverby stuff
// external array vs internal
// optimised allpass chain, shared write pointer


// OK there is some way around the union problem by making union members static
// wait thats if i want to use them as template parameters
// i think they are actually something i want to do in constructor

// TODO check write/increment order... 
// TODO check modulo behaviour.. where was it breaking previously??
// TODO can i combine these into one class?


// delay line with internal buffer - usage: DLi<float, 32768> d0;
template<typename T, uint32_t length, uint32_t rate = 1, Interptype t = LINEAR>
class DLi{
private:
    uint32_t i = 0;
    uint32_t write_head = 0;
    T a[length] = {0};
public:
    inline void operator[](float in){
        i = (i - 1) % (length << (rate - 1));
        write_head = i >> (rate - 1);
        a[write_head] = signal_cast<T>(in);
    }

    inline float operator()(uint32_t dt){
        return signal_cast<float>(a[(write_head + dt) % length]);
    }

    inline float operator()(float dt){ // 0.0f <= dt <= 1.0f
        float fsamp = dt * (length - 1);
        uint32_t isamp = (uint32_t)(fsamp);
        float u = fsamp - isamp;
        float x = signal_cast(a[(write_head + isamp) % length]);
        float y = signal_cast(a[(write_head + isamp + 1) % length]);
        return interp_lin(x, y, u);
    }
};


// delay line with external buffer, extra memory indirection
// usage: DLe<float, 32768> d0{arrayname};
// can use union etc without making them static
template<typename T, uint32_t length, uint32_t rate = 1, Interptype t = LINEAR>
class DLe{
private:
    uint32_t i = 0;
    uint32_t write_head = 0;
    T (&a)[length];
public:
    constexpr DLe(T (&a_ext)[length]): a(a_ext){}

    inline void operator[](float in){
        i = (i - 1) % (length << (rate - 1));
        write_head = i >> (rate - 1);
        a[write_head] = signal_cast<T>(in);
    }

    inline float operator()(uint32_t dt){
        return signal_cast<float>(a[(write_head + dt) % length]);
    }

    inline float operator()(float dt){ // 0.0f <= dt <= 1.0f
        float fsamp = dt * (length - 1);
        uint32_t isamp = (uint32_t)(fsamp);
        float u = fsamp - isamp;
        float x = signal_cast(a[(write_head + isamp) % length]);
        float y = signal_cast(a[(write_head + isamp + 1) % length]);
        return interp_lin(x, y, u);
    }
};



template<uint32_t n, typename T = float, uint32_t rate = 1, typename DT = float>
class APDi {
private:
    DLi<T, n, rate> D;
public:
    float c;
    DT dt;

    APDi(float c_ext, DT dt_ext) : c(c_ext), dt(dt_ext){}
    inline void operator[](DT x){dt=x;} // set delay length

    inline float operator()(float in){
        float w = in - c*D(dt);
        D[w];
        return D(dt) + c*w;
    }
};



////////////////////////////////////////////////////////// env
// envelope follower, compressor/limiter
// slew limiter?
// decay env, AD, ADSR
// MSEG primitives - same as above?
// time/coefficient calculations, bendy phase distortion / env slope stuff


class Slope{
private:
    float s = 0.0f;
    float rise = 0.0f;
    float fall = 0.0f;
public:
    Slope(){}
    Slope(float r_ext, float f_ext) : rise(r_ext), fall(f_ext) {}
    inline void rf(float rise_ext, float fall_ext){ rise=rise_ext; fall=fall_ext; }
    inline float operator()(float in){
        return s += ((in-s)>0 ? rise : fall) * (in-s);
    }
};

class Limiter{
private:
	Slope envelope;
public:
	Limiter(float Rise, float Fall){
		envelope.rf(Rise, Fall);
	}
	Limiter(){}
	float operator()(float in){
		float smoothrect = envelope(fabsf(in));
		float gain = smoothrect <= 1.0f ? 1.0f : 1.0f / smoothrect;
		return gain * in;
	}
};

struct slew{
    float s = 0.0f;
    float rise = 0.0f;
    float fall = 0.0f;
    inline void rf(float rise_ext, float fall_ext){ rise=rise_ext; fall=fall_ext; }
    inline float operator()(float in){
        float error = in - s;
        if (error > rise) {error = rise;}
        else if (error < -fall) {error = -fall;}
        return s += error;
    }
};

struct ef{
    Slope sl;
    inline void ar(float a_ext, float r_ext){sl.rf(a_ext, r_ext);}
    inline float operator()(float in){
        return sl(fabsf(in));
    }
};


class Decay{
private:
	float s = 0.0f;
	float d = 0.9999f;
public:
	float operator()(){
		return s *= d;
	}
	void t(){
		s = 1.0f;
	}

};


////////////////////////////////////////////////////////// sample playback / granulorb
// windowing
// grains
// engine w buffer and grain handling logic


////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// fshift
// oversampling

// will it make sense to have a separate type per array?
// for external array i can deduce dimensions and auto& nttp array

template<typename T, int32_t wavelen, int32_t xlen , Interptype xtype, T (&w)[xlen][wavelen]>
struct wt {
    //T (&w)[xlen][wavelen];
    float X = 0.0f;
    //constexpr wt( T (&w_ext)[xlen][wavelen] ) : w(w_ext){}
    inline void wave(float X_ext){ X=X_ext; }
    inline float operator()(float U){

        // SS is not really a noticeable improvement for wave scanning, and is actually worse for phasor lookup
    	// weird though it should be...correlated xfade

        if (xtype == CUBIC){

            int32_t Uq = round_f2i(U*(wavelen));
            float u = U*(wavelen) - (float)Uq;

            int32_t Xq = round_f2i(X*(xlen-1));
            float x = X*(xlen-1) - (float)Xq;

            float x0 = signal_cast(w[(Xq  )&(xlen-1)][(Uq-1)&(wavelen-1)]);
            float x1 = signal_cast(w[(Xq  )&(xlen-1)][(Uq  )&(wavelen-1)]);
            float x2 = signal_cast(w[(Xq  )&(xlen-1)][(Uq+1)&(wavelen-1)]);
            float x3 = signal_cast(w[(Xq  )&(xlen-1)][(Uq+2)&(wavelen-1)]);

            float y0 = signal_cast(w[(Xq+1)&(xlen-1)][(Uq-1)&(wavelen-1)]);
            float y1 = signal_cast(w[(Xq+1)&(xlen-1)][(Uq  )&(wavelen-1)]);
            float y2 = signal_cast(w[(Xq+1)&(xlen-1)][(Uq+1)&(wavelen-1)]);
            float y3 = signal_cast(w[(Xq+1)&(xlen-1)][(Uq+2)&(wavelen-1)]);

            float wave1 = interp_cubic(x0, x1, x2, x3, u);
            float wave2 = interp_cubic(y0, y1, y2, y3, u);

			return interp_lin(wave1, wave2, x);

        } else {

            int32_t Uq = round_f2i(U*(wavelen));
            float u = U*(wavelen) - (float)Uq;

            int32_t Xq = round_f2i(X*(xlen-1));
            float x = X*(xlen-1) - (float)Xq;

			float wave1 = interp_lin(w[(Xq  )&(xlen-1)][(Uq  )&(wavelen-1)], w[(Xq  )&(xlen-1)][(Uq+1)&(wavelen-1)], u);
			float wave2 = interp_lin(w[(Xq+1)&(xlen-1)][(Uq  )&(wavelen-1)], w[(Xq+1)&(xlen-1)][(Uq+1)&(wavelen-1)], u);

			return interp_lin(wave1, wave2, x);
        }

    }
};



lut<float, 256, LINEAR> sinlut{
    0, 0.0245412, 0.0490677, 0.0735646, 0.0980171, 0.122411, 0.14673, 0.170962, 0.19509, 0.219101, 0.24298, 0.266713, 0.290285, 0.313682, 0.33689, 0.359895, 0.382683, 0.405241, 0.427555, 0.449611, 0.471397, 0.492898, 0.514103, 0.534998, 0.55557, 0.575808, 0.595699, 0.615232, 0.634393, 0.653173, 0.671559, 0.689541, 0.707107, 0.724247, 0.740951, 0.757209, 0.77301, 0.788346, 0.803208, 0.817585, 0.83147, 0.844854, 0.857729, 0.870087, 0.881921, 0.893224, 0.903989, 0.91421, 0.92388, 0.932993, 0.941544, 0.949528, 0.95694, 0.963776, 0.970031, 0.975702, 0.980785, 0.985278, 0.989177, 0.99248, 0.995185, 0.99729, 0.998795, 0.999699, 1, 0.999699, 0.998795, 0.99729, 0.995185, 0.99248, 0.989177, 0.985278, 0.980785, 0.975702, 0.970031, 0.963776, 0.95694, 0.949528, 0.941544, 0.932993, 0.92388, 0.91421, 0.903989, 0.893224, 0.881921, 0.870087, 0.857729, 0.844854, 0.83147, 0.817585, 0.803208, 0.788346, 0.77301, 0.757209, 0.740951, 0.724247, 0.707107, 0.689541, 0.671559, 0.653173, 0.634393, 0.615232, 0.595699, 0.575808, 0.55557, 0.534998, 0.514103, 0.492898, 0.471397, 0.449611, 0.427555, 0.405241, 0.382683, 0.359895, 0.33689, 0.313682, 0.290285, 0.266713, 0.24298, 0.219101, 0.19509, 0.170962, 0.146731, 0.122411, 0.0980171, 0.0735644, 0.0490677, 0.0245412, -8.74228e-08, -0.0245411, -0.0490677, -0.0735646, -0.098017, -0.122411, -0.14673, -0.170962, -0.19509, -0.219101, -0.24298, -0.266713, -0.290285, -0.313682, -0.33689, -0.359895, -0.382683, -0.405241, -0.427555, -0.449611, -0.471397, -0.492898, -0.514103, -0.534998, -0.55557, -0.575808, -0.595699, -0.615232, -0.634393, -0.653173, -0.671559, -0.689541, -0.707107, -0.724247, -0.740951, -0.757209, -0.77301, -0.788346, -0.803208, -0.817585, -0.831469, -0.844853, -0.857729, -0.870087, -0.881921, -0.893224, -0.903989, -0.91421, -0.92388, -0.932993, -0.941544, -0.949528, -0.95694, -0.963776, -0.970031, -0.975702, -0.980785, -0.985278, -0.989177, -0.99248, -0.995185, -0.99729, -0.998795, -0.999699, -1, -0.999699, -0.998795, -0.99729, -0.995185, -0.99248, -0.989177, -0.985278, -0.980785, -0.975702, -0.970031, -0.963776, -0.95694, -0.949528, -0.941544, -0.932993, -0.923879, -0.91421, -0.903989, -0.893224, -0.881921, -0.870087, -0.857729, -0.844853, -0.83147, -0.817585, -0.803208, -0.788346, -0.77301, -0.757209, -0.740951, -0.724247, -0.707107, -0.689541, -0.671559, -0.653173, -0.634393, -0.615231, -0.595699, -0.575808, -0.55557, -0.534998, -0.514103, -0.492898, -0.471397, -0.449612, -0.427555, -0.405241, -0.382683, -0.359895, -0.33689, -0.313682, -0.290285, -0.266713, -0.24298, -0.219101, -0.19509, -0.170962, -0.14673, -0.122411, -0.0980172, -0.0735646, -0.0490676, -0.0245411
};


struct fshift{
    float AS[6] = {0.999646f, 0.997312f, 0.989106f, 0.957003f, 0.837943f, 0.436841f}; //im
    float AC[6] = {0.998771f, 0.994549f, 0.978303f, 0.915717f, 0.696257f, -0.152462f}; //re
    //lut<float> & S;
    Phasor p;
    float inc = 0.0f;
    Ap1 as1{AS[0]};
    Ap1 as2{AS[1]};
    Ap1 as3{AS[2]};
    Ap1 as4{AS[3]};
    Ap1 as5{AS[4]};
    Ap1 as6{AS[5]};
    Ap1 ac1{AC[0]};
    Ap1 ac2{AC[1]};
    Ap1 ac3{AC[2]};
    Ap1 ac4{AC[3]};
    Ap1 ac5{AC[4]};
    Ap1 ac6{AC[5]};
    
    float up, down;

    //fshift(lut<float> & S_ext): S(S_ext){}

    inline void operator[](float inc_ext){inc=inc_ext;}

    inline float operator()(float in){
        float x1 = sinlut(p(inc))       * as6(as5(as4(as3(as2(as1(in))))));
        float x2 = sinlut(p(inc)+0.25f) * ac6(ac5(ac4(ac3(ac2(ac1(in))))));

        down = 0.67f*(x1 + x2); // approx /sqrt(2)
        return up = 0.67f*(x1 - x2); // ??????? check polarity etc...sum = up, difference=down???? or other way??
    }

};




struct fourpole{

	Tpt1<LP1> pole1;
	Tpt1<LP1> pole2;
	Tpt1<LP1> pole3;
	Tpt1<LP1> pole4;
	float f_out = 0.0f;

	float cutoff = 0.5f;
	float reso = 0.0f;


	inline float operator()(float in){

		float c = clip_unipolar(cutoff);
		pole1.setG(0.97f*c);
		pole2.setG(c);
		pole3.setG(0.91f*c);
		pole4.setG(c);

		float insum = fclip(in - 5.0f*reso*f_out);
		f_out = pole4(pole3(pole2(pole1( insum ))));

		return f_out*(1.0f + reso);

	}

	inline void reset(){
		pole1.s = 0.0f;
		pole2.s = 0.0f;
		pole3.s = 0.0f;
		pole4.s = 0.0f;
		f_out = 0.0f;
	}
};