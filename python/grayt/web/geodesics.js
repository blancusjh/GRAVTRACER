/* Float64 browser preview of the CPU Hamiltonian ray tracer.
 * State: (t,r,theta,phi,p_r,p_theta), conserved (p_t,p_phi).
 * RKDP45 with FSAL, RMS error control, and quartic dense event location.
 * Analytic Kerr/q expressions are generated from gpu/kernel.cl. The RN
 * comparison uses the exact analytic exterior, rather than a metric table.
 * No atmosphere, scattering, disk self-gravity, or spectral color model.
 */
globalThis.GraytRayTracer = class {
  constructor(model) {
    this.model = model;
    const vector = n => new Float64Array(n);
    this.y = vector(6); this.next = vector(6); this.tmp = vector(6);
    this.hit = vector(6); this.error = vector(6);
    this.k = Array.from({length: 7}, () => vector(6));
    this.dense = Array.from({length: 5}, () => vector(6));
    this.gd = vector(5); this.gu = vector(5); this.dr = vector(5); this.dt = vector(5);
    this.A = [
      [1/5], [3/40,9/40], [44/45,-56/15,32/9],
      [19372/6561,-25360/2187,64448/6561,-212/729],
      [9017/3168,-355/33,46732/5247,49/176,-5103/18656],
      [35/384,0,500/1113,125/192,-2187/6784,11/84],
    ];
    this.E = [35/384-5179/57600,0,500/1113-7571/16695,
      125/192-393/640,-2187/6784+92097/339200,11/84-187/2100,-1/40];
    this.D = [-12715105075/11282082432,0,87487479700/32700410799,
      -10690763975/1880347072,701980252875/199316789632,
      -1453857185/822651844,69997945/29380423];
  }

  cov(r, th) {
    const {kind, parameter: p} = this.model;
    if (kind === 'rn') {
      const f = 1-2/r+p*p/(r*r), s = Math.sin(th);
      this.gd.set([-f,0,1/f,r*r,r*r*s*s]);
    } else if (kind === 'q') {
      GraytMetrics.q_cov(p,r,th,this.gd);
    } else {
      GraytMetrics.kerr_cov(p,r,th,this.gd);
    }
    return this.gd;
  }

  contra(r, th) {
    const {kind, parameter: p} = this.model;
    const {gu,dr,dt} = this;
    if (kind === 'rn') {
      const s = Math.sin(th), c = Math.cos(th);
      const f = 1-2/r+p*p/(r*r), fp = 2/(r*r)-2*p*p/(r*r*r);
      gu.set([-1/f,0,f,1/(r*r),1/(r*r*s*s)]);
      dr.set([fp/(f*f),0,fp,-2/(r*r*r),-2/(r*r*r*s*s)]);
      dt.set([0,0,0,0,-2*c/(r*r*s*s*s)]);
    } else if (kind === 'q') {
      GraytMetrics.q_contra(p,r,th,gu,dr,dt);
    } else {
      GraytMetrics.kerr_contra(p,r,th,gu,dr,dt);
    }
  }

  rhs(y, out) {
    this.contra(y[1],y[2]);
    const {gu,dr,dt,pt,pp} = this, pr=y[4], pth=y[5];
    out[0]=gu[0]*pt+gu[1]*pp; out[1]=gu[2]*pr;
    out[2]=gu[3]*pth; out[3]=gu[1]*pt+gu[4]*pp;
    out[4]=-0.5*(pt*pt*dr[0]+2*pt*pp*dr[1]+pr*pr*dr[2]+pth*pth*dr[3]+pp*pp*dr[4]);
    out[5]=-0.5*(pt*pt*dt[0]+2*pt*pp*dt[1]+pr*pr*dt[2]+pth*pth*dt[3]+pp*pp*dt[4]);
  }

  constraint(y) {
    this.contra(y[1],y[2]);
    const {gu,pt,pp}=this;
    return 0.5*(gu[0]*pt*pt+2*gu[1]*pt*pp+gu[2]*y[4]**2+gu[3]*y[5]**2+gu[4]*pp*pp);
  }

  step(h, valid) {
    const {y,k,tmp,next,A,E,error} = this;
    if (!valid) this.rhs(y,k[0]);
    for (let stage=1;stage<7;stage++) {
      const out=stage===6?next:tmp, weights=A[stage-1];
      for (let i=0;i<6;i++) {
        let sum=0;
        for (let j=0;j<stage;j++) sum+=weights[j]*k[j][i];
        out[i]=y[i]+h*sum;
      }
      this.rhs(out,k[stage]);
    }
    for (let i=0;i<6;i++) {
      let sum=0;
      for (let j=0;j<7;j++) sum+=E[j]*k[j][i];
      error[i]=h*sum;
    }
  }

  refine(h, sphere=null) {
    const {y,next,k,dense:rc,D,hit,tmp}=this;
    for (let i=0;i<6;i++) {
      rc[0][i]=y[i]; rc[1][i]=next[i]-y[i];
      rc[2][i]=h*k[0][i]-rc[1][i];
      rc[3][i]=rc[1][i]-h*k[6][i]-rc[2][i];
      let sum=0;
      for (let j=0;j<7;j++) sum+=D[j]*k[j][i];
      rc[4][i]=h*sum;
    }
    const value=v=>sphere===null?Math.cos(v[2]):v[1]-sphere;
    const s0=value(y);
    let lo=0, hi=1;
    hit.set(next);
    for (let j=0;j<48;j++) {
      const t=(lo+hi)/2, t1=1-t;
      for (let i=0;i<6;i++) tmp[i]=rc[0][i]+t*(rc[1][i]+t1*(rc[2][i]+t*(rc[3][i]+t1*rc[4][i])));
      if (s0*value(tmp)<=0) { hi=t; hit.set(tmp); } else lo=t;
    }
    return hit;
  }

  trace(screenX, screenY, camera, options={}) {
    const r=camera.r, th=camera.theta*Math.PI/180, ph=camera.phi*Math.PI/180;
    // Match render_geometry's reflection of the horizontal screen axis.
    const x=-screenX, gd=this.cov(r,th);
    const at=Math.sqrt(gd[4]/(gd[1]*gd[1]-gd[0]*gd[4]));
    this.pt=1/at-x*gd[1]/(r*Math.sqrt(gd[4]));
    this.pp=-x*Math.sqrt(gd[4])/r;
    this.y.set([0,r,th,ph,Math.sqrt(gd[2]*Math.max(0,1-(x*x+screenY*screenY)/(r*r))),screenY*Math.sqrt(gd[3])/r]);
    const rtol=options.rtol??1e-7, atol=options.atol??1e-9;
    const escape=options.escape??200, budget=options.maxSteps??20000;
    const {model,y,next,k,error}=this;
    let h=-1, valid=false, status=3, herr=0, n=0;
    for (;n<budget;n++) {
      let accepted=false, used=h;
      for (let retry=0;retry<60;retry++) {
        this.step(h,valid); valid=true;
        let err2=0;
        for (let i=0;i<6;i++) {
          const e=error[i]/(atol+rtol*Math.max(Math.abs(y[i]),Math.abs(next[i])));
          err2+=e*e;
        }
        const err=Math.max(Math.sqrt(err2/6),1e-30);
        if (!Number.isFinite(err) || err>1) {
          h*=Number.isFinite(err)?Math.max(0.2,0.9*err**(-0.25)):0.2;
          if (Math.abs(h)<1e-12) break;
          continue;
        }
        used=h;
        h*=Math.min(5,Math.max(0.2,0.9*err**(-0.2)));
        h=-Math.max(1e-12,Math.min(Math.abs(h),Math.min(100,Math.max(25,0.1*next[1]))));
        accepted=true; break;
      }
      if (!accepted) break;
      if (options.monitor) herr=Math.max(herr,Math.abs(this.constraint(next)));
      if (model.disk && Math.cos(y[2])*Math.cos(next[2])<0) {
        const hit=this.refine(used);
        if (hit[1]>=model.disk.rIn && hit[1]<=model.disk.rOut) {
          y.set(hit); status=2; break;
        }
      }
      if (next[1]<=model.capture) {
        y.set(this.refine(used,model.capture)); status=1; break;
      }
      if (next[1]>=escape) {
        y.set(this.refine(used,escape)); status=0; break;
      }
      y.set(next); k[0].set(k[6]);
    }
    return {status,y:y.slice(),pt:this.pt,pphi:this.pp,steps:n+1,herr};
  }

  emission(ray) {
    const {model}=this, r=ray.y[1];
    if (ray.status===2 && model.disk) {
      const omega=model.kind==='rn'?Math.sqrt(1/r**3-model.parameter**2/r**4):1/(r**1.5+model.parameter);
      const gd=this.cov(r,Math.PI/2);
      const g=Math.sqrt(-gd[0]-2*omega*gd[1]-omega*omega*gd[4])/(ray.pt+omega*ray.pphi);
      const d=model.disk, coordinate=(r-d.rIn)/(d.rOut-d.rIn)*(d.flux.length-1);
      const index=Math.max(0,Math.min(d.flux.length-2,Math.floor(coordinate)));
      const weight=Math.max(0,Math.min(1,coordinate-index));
      return g**4*((1-weight)*d.flux[index]+weight*d.flux[index+1]);
    }
    if (ray.status===1 && model.surface) {
      return (Math.sqrt(-this.cov(r,ray.y[2])[0])/ray.pt)**4;
    }
    return 0;
  }
};
