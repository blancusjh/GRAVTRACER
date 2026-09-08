/* Each worker traces a row stripe, then caches the vacuum geometry.
 * At fixed r/theta/field of view, axial symmetry permits continuous phi
 * changes without retracing. The celestial map and any surface hot spots
 * are evaluated at the rotated endpoint. No image or trajectory morphing.
 */
let settings, sky, tracer, currentModel=-1, revision=0;
const cache=new Map();
const pause=()=>new Promise(resolve=>setTimeout(resolve,0));

function skyColor(th, ph, out, offset, brightness) {
  const h=settings.skyHeight,w=settings.skyWidth;
  const canonical=Math.acos(Math.max(-1,Math.min(1,Math.cos(th))));
  const longitude=Math.atan2(Math.sin(th)*Math.sin(ph),Math.sin(th)*Math.cos(ph));
  const x=(((longitude/(2*Math.PI))%1+1)%1)*w-0.5;
  const y=Math.max(0,Math.min(h-1,canonical/Math.PI*h-0.5));
  const ix=Math.floor(x),iy=Math.floor(y),sx=x-ix,sy=y-iy;
  const x0=(ix+w)%w,x1=(ix+1+w)%w,y1=Math.min(h-1,iy+1);
  for(let c=0;c<3;c++) {
    const top=(1-sx)*sky[4*(iy*w+x0)+c]+sx*sky[4*(iy*w+x1)+c];
    const bottom=(1-sx)*sky[4*(y1*w+x0)+c]+sx*sky[4*(y1*w+x1)+c];
    out[offset+c]=brightness*((1-sy)*top+sy*bottom);
  }
}

function colorize(map, request, model) {
  const rgb=new Uint8ClampedArray(map.length/5*4), shift=request.camera.phi*Math.PI/180;
  for(let p=0;p<map.length/5;p++) {
    const status=map[5*p],th=map[5*p+1],ph=map[5*p+2]+shift;
    let value=map[5*p+3];
    if(status===0 && request.skyBrightness>0) {
      skyColor(th,ph,rgb,4*p,request.skyBrightness);
    } else if(status===3) {
      rgb[4*p]=255;rgb[4*p+2]=255;
    } else if(status===2 || (status===1 && model.surface)) {
      if(status===1) {
        const dot=Math.cos(th)*Math.cos(0.8)+Math.sin(th)*Math.sin(0.8)*Math.cos(ph-0.5);
        value*=model.surface==='spots'?3e-5+8e-4*Math.exp(-(1-dot)/0.035)+4e-4*Math.exp(-(1+dot)/0.035):3e-4;
      }
      const index=Math.min(255,Math.floor(256*(-Math.expm1(-request.exposure*value))));
      for(let c=0;c<3;c++) rgb[4*p+c]=settings.palette[3*index+c];
    }
    rgb[4*p+3]=255;
  }
  return rgb;
}

async function render(request) {
  const token=++revision, model=settings.models[request.model];
  const {camera,width,height,row0,row1}=request;
  // All supported metrics and disk emission models are axisymmetric.
  const key=JSON.stringify([request.model,camera.r,camera.theta,camera.zoom,width,height,request.rtol,row0,row1]);
  let map=cache.get(key), failed=0;
  if(!map) {
    if(currentModel!==request.model) {
      tracer=new GraytRayTracer(model);currentModel=request.model;
    }
    map=new Float64Array(width*(row1-row0)*5);
    const traceCamera={...camera,phi:0};
    for(let row=row0;row<row1;row++) {
      for(let col=0;col<width;col++) {
        const x=(-26+52*(col+0.5)/width)/camera.zoom;
        const y=(16-32*(row+0.5)/height)/camera.zoom;
        const ray=tracer.trace(x,y,traceCamera,{rtol:request.rtol,atol:request.rtol/100});
        const p=((row-row0)*width+col)*5;
        map[p]=ray.status;map[p+1]=ray.y[2];map[p+2]=ray.y[3];
        map[p+3]=tracer.emission(ray);map[p+4]=ray.y[1];
      }
      if((row-row0)%4===3) {
        self.postMessage({type:'progress',id:request.id,worker:request.worker,done:(row-row0+1)/(row1-row0)});
        await pause();
        if(token!==revision) return;
      }
    }
    if(token!==revision) return;
    cache.set(key,map);
    // Keep the last full view and preview; rotations only change the sky.
    if(cache.size>3) cache.delete(cache.keys().next().value);
  }
  for(let p=0;p<map.length;p+=5) if(map[p]===3) failed++;
  const rgb=colorize(map,request,model);
  self.postMessage({type:'frame',id:request.id,worker:request.worker,
    row0,row1,width,height,failed,buffer:rgb.buffer},[rgb.buffer]);
}

self.onmessage=event=>{
  const message=event.data;
  if(message.type==='init') {settings=message.settings;sky=message.sky;return;}
  if(message.type==='cancel') {revision++;return;}
  render(message).catch(error=>self.postMessage({type:'error',id:message.id,message:error.message}));
};
