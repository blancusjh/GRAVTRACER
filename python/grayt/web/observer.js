/* Offline observer controls. Geometry runs in cancellable Web Workers;
 * only complete views replace the canvas, so different cameras never mix.
 */
(async () => {
  const root=document.getElementById('observer-view');
  if(!root) return;
  const config=JSON.parse(document.getElementById('observer-settings').textContent);
  const source=JSON.parse(document.getElementById('observer-worker-source').textContent);
  const byId=id=>document.getElementById('observer-'+id);
  const canvas=byId('canvas'), context=canvas.getContext('2d');
  const modelSelect=byId('model'), status=byId('status');
  const workers=[];
  let workerURL, job=null, sequence=0, refineTimer, gesture=null, playing=false;
  let last=null, lastStamp=0, scheduled=0, orbitAnimation=0;
  const camera={r:100,theta:85,phi:0,zoom:1};
  let selected=0;
  const clamp=(x,lo,hi)=>Math.min(hi,Math.max(lo,x));
  const geometryKey=()=>JSON.stringify([selected,camera.r,camera.theta,camera.zoom]);

  function updateControls() {
    byId('inclination').value=camera.theta;
    byId('azimuth').value=camera.phi;
    byId('inclination-value').textContent=camera.theta.toFixed(1)+'°';
    byId('azimuth-value').textContent=camera.phi.toFixed(1)+'°';
    byId('position').textContent=`i ${camera.theta.toFixed(1)}° · φ ${camera.phi.toFixed(1)}° · ${camera.zoom.toFixed(2)}×`;
    const model=config.models[selected];
    byId('note').textContent=model.note;
    byId('reference').href=model.reference;
    byId('reference').textContent='Open reference render';
  }

  function finish(message) {
    if(!job || message.id!==job.id) return;
    if(message.type==='error') {
      status.textContent='Rendering error: '+message.message;
      root.dataset.rendering='false';job=null;stop();return;
    }
    if(message.type==='progress') {
      job.progress[message.worker]=message.done;
      const done=job.progress.reduce((a,b)=>a+b,0)/workers.length;
      status.textContent=`${job.samples>1?'Refining':'Tracing'} view… ${Math.round(done*100)}%`;
      return;
    }
    if(message.type!=='frame') return;
    job.data.set(new Uint8ClampedArray(message.buffer),message.row0*job.width*4);
    job.received++;job.failed+=message.failed;
    if(job.received!==workers.length) return;
    const {width,height,samples,data}=job, nx=width/samples,ny=height/samples;
    const averaged=new Uint8ClampedArray(nx*ny*4);
    for(let y=0;y<ny;y++) for(let x=0;x<nx;x++) {
      const pixel=(y*nx+x)*4;
      for(let c=0;c<3;c++) {
        let sum=0;
        for(let sy=0;sy<samples;sy++) for(let sx=0;sx<samples;sx++) {
          sum+=data[((y*samples+sy)*width+x*samples+sx)*4+c];
        }
        averaged[pixel+c]=sum/(samples*samples);
      }
      averaged[pixel+3]=255;
    }
    canvas.width=nx;canvas.height=ny;
    context.putImageData(new ImageData(averaged,nx,ny),0,0);
    byId('poster').hidden=true;
    last={key:job.key,width,height,samples,rtol:job.rtol,camera:job.camera,model:job.model};
    byId('save').disabled=false;
    status.textContent=job.failed?`${job.failed} unresolved rays shown in magenta. Try another viewpoint.`:
      (samples>1?'Ready · drag to orbit, scroll to zoom':'Preview · release to refine');
    root.dataset.rendering='false';root.dataset.lastFrame=String(job.id);
    root.dataset.failedRays=String(job.failed);
    root.dataset.model=config.models[selected].id;
    root.dataset.theta=String(job.camera.theta);root.dataset.phi=String(job.camera.phi);
    job=null;
    if(playing) scheduleOrbit();
  }

  function render(refine=false) {
    if(!workers.length) return;
    if(refine && scheduled) {cancelAnimationFrame(scheduled);scheduled=0;}
    const key=geometryKey();
    const reuse=last?.key===key?last:null;
    const detailed=byId('quality').value==='detail';
    let width=96,height=60,samples=1,rtol=1e-6;
    if(refine) {width=detailed?960:640;height=detailed?600:400;samples=2;rtol=1e-8;}
    else if(reuse) ({width,height,samples,rtol}=reuse);
    job={id:++sequence,key,width,height,samples,rtol,camera:{...camera},model:selected,
      data:new Uint8ClampedArray(width*height*4),received:0,failed:0,
      progress:Array(workers.length).fill(0)};
    root.dataset.rendering='true';
    if(!reuse) status.textContent=refine?'Refining view…':'Tracing preview…';
    workers.forEach((worker,index)=>worker.postMessage({
      type:'render',id:job.id,worker:index,model:selected,camera:{...camera},
      width,height,rtol,row0:Math.floor(height*index/workers.length),
      row1:Math.floor(height*(index+1)/workers.length),
      exposure:Number(byId('exposure').value),skyBrightness:Number(byId('sky').value),
    }));
  }

  function changed() {
    stop();updateControls();clearTimeout(refineTimer);
    job=null;root.dataset.rendering='true';status.textContent='Updating view…';
    // Coalesce mouse events into one request per animation frame.
    if(!scheduled) {
      scheduled=requestAnimationFrame(()=>{scheduled=0;render(false);});
    }
    refineTimer=setTimeout(()=>render(true),350);
  }

  function stop() {
    playing=false;byId('play').textContent='Auto orbit';
    if(orbitAnimation) {cancelAnimationFrame(orbitAnimation);orbitAnimation=0;}
    byId('play').setAttribute('aria-pressed','false');
  }

  function scheduleOrbit() {
    if(orbitAnimation) return;
    orbitAnimation=requestAnimationFrame(stamp=>{
      orbitAnimation=0;
      if(!playing) return;
      const delta=Math.min(0.1,(stamp-lastStamp)/1000);lastStamp=stamp;
      camera.phi=(camera.phi+delta*12)%360;
      updateControls();render(false);
    });
  }

  try {
    const skyImage=new Image();skyImage.src=config.skyURL;await skyImage.decode();
    const scratch=document.createElement('canvas');
    scratch.width=skyImage.width;scratch.height=skyImage.height;
    const scratchContext=scratch.getContext('2d');scratchContext.drawImage(skyImage,0,0);
    const pixels=scratchContext.getImageData(0,0,scratch.width,scratch.height).data;
    workerURL=URL.createObjectURL(new Blob([source],{type:'text/javascript'}));
    const count=Math.max(1,Math.min(4,navigator.hardwareConcurrency||2));
    for(let i=0;i<count;i++) {
      const worker=new Worker(workerURL);
      worker.onmessage=event=>finish(event.data);
      worker.onerror=event=>{
        status.textContent='Browser worker failed: '+event.message;
        root.dataset.rendering='false';stop();
      };
      worker.postMessage({type:'init',settings:{models:config.models,palette:config.palette,
        skyWidth:scratch.width,skyHeight:scratch.height},sky:pixels});
      workers.push(worker);
    }
  } catch(error) {
    status.textContent='The interactive renderer could not start: '+error.message;
    return;
  }

  config.models.forEach((model,index)=>{
    const option=new Option(model.label,String(index));modelSelect.add(option);
  });
  modelSelect.addEventListener('change',()=>{selected=Number(modelSelect.value);changed();});
  byId('inclination').addEventListener('input',event=>{camera.theta=Number(event.target.value);changed();});
  byId('azimuth').addEventListener('input',event=>{camera.phi=Number(event.target.value);changed();});
  for(const id of ['exposure','sky','quality']) byId(id).addEventListener('input',changed);
  canvas.addEventListener('pointerdown',event=>{
    if(event.button!==0) return;
    stop();clearTimeout(refineTimer);canvas.focus({preventScroll:true});
    canvas.setPointerCapture(event.pointerId);gesture={x:event.clientX,y:event.clientY};
  });
  canvas.addEventListener('pointermove',event=>{
    if(!gesture) return;
    camera.phi=(camera.phi-(event.clientX-gesture.x)*0.25+360)%360;
    camera.theta=clamp(camera.theta+(event.clientY-gesture.y)*0.2,5,175);
    gesture={x:event.clientX,y:event.clientY};changed();clearTimeout(refineTimer);
  });
  const endGesture=()=>{if(gesture) {gesture=null;clearTimeout(refineTimer);render(true);}};
  canvas.addEventListener('pointerup',endGesture);
  canvas.addEventListener('pointercancel',endGesture);
  canvas.addEventListener('lostpointercapture',endGesture);
  canvas.addEventListener('wheel',event=>{
    event.preventDefault();camera.zoom=clamp(camera.zoom*Math.exp(-event.deltaY*0.001),0.65,4);changed();
  },{passive:false});
  canvas.addEventListener('keydown',event=>{
    const actions={ArrowLeft:()=>camera.phi=(camera.phi+5)%360,
      ArrowRight:()=>camera.phi=(camera.phi+355)%360,
      ArrowUp:()=>camera.theta=clamp(camera.theta-2,5,175),
      ArrowDown:()=>camera.theta=clamp(camera.theta+2,5,175),
      '+':()=>camera.zoom=clamp(camera.zoom*1.1,0.65,4),
      '-':()=>camera.zoom=clamp(camera.zoom/1.1,0.65,4)};
    if(actions[event.key]) {event.preventDefault();actions[event.key]();changed();}
  });
  byId('reset').addEventListener('click',()=>{
    Object.assign(camera,{theta:85,phi:0,zoom:1});changed();
  });
  byId('play').addEventListener('click',()=>{
    if(playing) {stop();render(true);return;}
    playing=true;clearTimeout(refineTimer);lastStamp=performance.now();
    byId('play').textContent='Pause orbit';byId('play').setAttribute('aria-pressed','true');
    if(!job) scheduleOrbit();
  });
  byId('save').addEventListener('click',()=>{
    if(!last) return;
    const link=document.createElement('a');
    const shown=last.camera;
    link.download=`${config.models[last.model].id}_i${shown.theta.toFixed(1)}_phi${shown.phi.toFixed(1)}.png`;
    link.href=canvas.toDataURL('image/png');link.click();
  });
  document.addEventListener('click',event=>{
    const link=event.target.closest('[data-observer-name]');
    if(!link) return;
    const name=link.dataset.observerName;
    const index=config.models.findIndex(model=>model.aliases.includes(name));
    if(index<0) return;
    selected=index;modelSelect.value=String(index);
    const angle=name.match(/_i(\d+)$/);
    camera.theta=angle?Number(angle[1]):70;camera.phi=0;camera.zoom=1;changed();
  });
  window.addEventListener('pagehide',event=>{
    if(event.persisted) return;
    workers.forEach(worker=>worker.terminate());URL.revokeObjectURL(workerURL);
  });
  updateControls();render(false);refineTimer=setTimeout(()=>render(true),1000);
})();
