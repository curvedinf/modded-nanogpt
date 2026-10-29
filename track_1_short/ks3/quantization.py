"""Adapted from curvedinf/k-splanifolds ea7a9c1, CC-BY-SA-4.0; see LICENSE.k-splanifolds."""
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice

@tr.jit
def _half_scale(maximum):
    # Round BEFORE quantizing so encoding and decoding use the same FP16 scale.
    return tl.minimum(tl.maximum(maximum/127., 2.**-24), 65504.).to(tl.float16).to(tl.float32)


@tr.jit
def _spine_features(x, mean, c0, c1, c2, c3, c4, c5, ps, scale, shift, LUT,
                    valid, NL: tl.constexpr):
    a=2*c0-2*c1+c2+c3;b=-3*c0+3*c1-2*c2-c3
    if NL==2:
        # INT8 tanh inputs/outputs, INT32 fixed-point Horner accumulators.
        # Q6 guard bits keep worst-case polynomial products within INT32.
        ti=libdevice.nearbyint(tl.minimum(tl.maximum(mean*.25,-3.2),3.2)*(127./3.2)).to(tl.int32)
        tq=tl.load(LUT+ti+128).to(tl.int32)
        warp=scale[None,:]*(tq.to(tl.float32)*(4./127.)+shift[None,:])
        zi=libdevice.nearbyint(tl.minimum(tl.maximum(warp,-3.2),3.2)*(127./3.2)).to(tl.int32)
        zq=tl.load(LUT+zi+128).to(tl.int32)
        poly=a[None,:].to(tl.int32)*64
        poly=_round_div127(poly*tq*4)+b[None,:].to(tl.int32)*64
        poly=_round_div127(poly*tq*4)+c2[None,:].to(tl.int32)*64
        poly=_round_div127(poly*tq*4)+c0[None,:].to(tl.int32)*64
        warped=4*c5[None,:].to(tl.int32)*64
        warped=_round_div127(warped*zq)+2*c4[None,:].to(tl.int32)*64
        warped=_round_div127(warped*zq)-3*c5[None,:].to(tl.int32)*64
        warped=_round_div127(warped*zq)-c4[None,:].to(tl.int32)*64
        h=x+(poly+warped).to(tl.float32)*(ps[None,:]/64.)
        h=tl.where(valid[None,:],h,0.)
        pre_scale=_half_scale(tl.max(tl.abs(h),1))
        hq=tl.minimum(tl.maximum(libdevice.nearbyint(h/pre_scale[:,None]),-127.),127.).to(tl.int32)
        hq=tl.maximum(hq,0);h=(hq*hq).to(tl.float32)*(pre_scale[:,None]*pre_scale[:,None])
    elif NL==1:
        th=(mean*.25).to(tl.float16)
        t=(4*libdevice.tanh(th.to(tl.float32))).to(tl.float16)
        warp=(scale[None,:]*(t.to(tl.float32)+shift[None,:])).to(tl.float16)
        z=libdevice.tanh(warp.to(tl.float32)).to(tl.float16)
        # Each polynomial stage rounds to FP16; tanh uses FP32 internally.
        poly=(a[None,:].to(tl.float16)*t+b[None,:].to(tl.float16)).to(tl.float16)
        poly=(poly*t+c2[None,:].to(tl.float16)).to(tl.float16)
        poly=(poly*t+c0[None,:].to(tl.float16)).to(tl.float16)
        warped=(4*c5[None,:].to(tl.float16)*z+2*c4[None,:].to(tl.float16)).to(tl.float16)
        warped=(warped*z-3*c5[None,:].to(tl.float16)).to(tl.float16)
        warped=(warped*z-c4[None,:].to(tl.float16)).to(tl.float16)
        h=tl.maximum((x+(poly.to(tl.float32)+warped.to(tl.float32))*ps[None,:]).to(tl.float16),0.)
        h=(h*h).to(tl.float16).to(tl.float32)
    else:
        t=4*libdevice.tanh(mean*.25);z=libdevice.tanh(scale[None,:]*(t+shift[None,:]))
        poly=((a[None,:]*t+b[None,:])*t+c2[None,:])*t+c0[None,:]
        warped=((4*c5[None,:]*z+2*c4[None,:])*z-3*c5[None,:])*z-c4[None,:]
        h=tl.maximum(x+(poly+warped)*ps[None,:],0.);h=h*h
    return tl.where(valid[None,:],h,0.)


@tr.jit
def _round_div127(value):
    # Triton integer // truncates toward zero. Correct both signs explicitly.
    quotient=value//127
    remainder=value-quotient*127
    return quotient+tl.where(remainder>63,1,tl.where(remainder < -63,-1,0))

