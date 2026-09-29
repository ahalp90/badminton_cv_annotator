'use strict';
const cases=__GALLERY_DATA__;
const choiceLabels={primary:'Bounded · weight 0.04 · overrun 4 px',weight_02:'Bounded · weight 0.02 · overrun 4 px',weight_08:'Bounded · weight 0.08 · overrun 4 px',overrun_02:'Bounded · weight 0.04 · overrun 2 px',overrun_08:'Bounded · weight 0.04 · overrun 8 px',old_fixed:'Old unrestricted net choice'};
const overrun={primary:4,weight_02:4,weight_08:4,overrun_02:2,overrun_08:8,old_fixed:null};
const cornerLabels=['Upper left','Upper right','Lower right','Lower left'];
const cropWidth=120,cropHeight=90;
const states=[],redraw=[];
let spaceHeld=false;
const byId=id=>document.getElementById(id);
function outline(corners){return corners.map((point,index)=>[point,corners[(index+1)%4]]);}
function strokeLines(ctx,lines,colour,thickness,scale,dashes=[]){
 ctx.save();ctx.setLineDash(dashes.map(value=>value*scale));ctx.beginPath();
 for(const [start,end] of lines){ctx.moveTo(...start);ctx.lineTo(...end);}
 ctx.strokeStyle='#071019';ctx.lineWidth=(thickness+3)*scale;ctx.stroke();
 ctx.strokeStyle=colour;ctx.lineWidth=thickness*scale;ctx.stroke();ctx.restore();
}
function marker(ctx,point,colour,scale,shape='circle'){
 ctx.save();ctx.setLineDash([]);ctx.beginPath();
 if(shape==='square')ctx.rect(point[0]-3*scale,point[1]-3*scale,6*scale,6*scale);
 else ctx.arc(...point,3.5*scale,0,Math.PI*2);
 ctx.fillStyle=colour;ctx.strokeStyle='#071019';ctx.lineWidth=1.5*scale;ctx.fill();ctx.stroke();ctx.restore();
}
function selectedKey(data,name){return name==='saved'?data.baseline_key:byId('view').value==='old_fixed'?data.old_fixed_trial_key:data.choices[byId('view').value];}
function draw(canvas,image,data,name,focus,isCrop){
 const ctx=canvas.getContext('2d');
 const box=isCrop?[focus[0]-cropWidth/2,focus[1]-cropHeight/2,cropWidth,cropHeight]:[0,0,...data.size];
 const sx=canvas.width/box[2],sy=canvas.height/box[3];
 ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle='#050b10';ctx.fillRect(0,0,canvas.width,canvas.height);
 ctx.imageSmoothingEnabled=!isCrop||byId('smooth').checked;
 ctx.setTransform(sx,0,0,sy,-box[0]*sx,-box[1]*sy);ctx.drawImage(image,0,0,...data.size);
 if(!byId('outlines').checked||spaceHeld)return;
 const key=selectedKey(data,name),view=data.views[key];
 if(!view)return;
 const thickness=Number(byId('width').value),scale=box[2]/canvas.getBoundingClientRect().width;
 ctx.globalAlpha=Number(byId('opacity').value);ctx.lineJoin='round';
 const geometry=view.geometry,mode=byId('mode').value;
 const lines=mode==='edges'?geometry.edges.slice():outline(geometry.corners);
 if(mode==='centres')lines.push(...geometry.centres);
 strokeLines(ctx,lines,name==='saved'?'#f5f5f0':'#00e5ff',thickness,scale);
 if(byId('projected').checked&&view.net.state==='measured'){
  strokeLines(ctx,view.net.pieces_working_px,'#ffb547',Math.max(1.5,thickness*.7),scale,[10,5]);
 }
 if(byId('base').checked&&view.row.net_state==='measured'){
  for(const feature of Object.values(view.row.posts)){
   const offset=feature.lowest_endpoint_offset_working_px;
   const threshold=name==='saved'?4:(overrun[byId('view').value]??4);
   const accepted=offset!==null&&offset>=-threshold;
   const colour=accepted?'#a7bfff':'#ffbd67';
   const segments=feature.covering_ids.map(index=>data.segments[index]).filter(Boolean);
   strokeLines(ctx,segments.map(segment=>[segment.slice(0,2),segment.slice(2,4)]),colour,2,scale,accepted?[]:[4,5]);
   feature.samples_working_px.forEach((point,index)=>marker(ctx,point,feature.sample_covered[index]?'#a7bfff':'#f5f5f0',scale,feature.sample_covered[index]?'square':'circle'));
   marker(ctx,feature.samples_working_px[0],'#ffb547',scale,'square');
  }
 }
 ctx.globalAlpha=1;
}
function facts(data,key,name){
 const view=data.views[key];if(!view)return 'No accepted full-court choice · no outline';
 const row=view.row,setting=byId('view').value;
 const score=name==='saved'?data.baseline_score:data.selected_scores[setting];
 const threshold=name==='saved'?4:(overrun[setting]??4);
 const supported=Object.values(row.posts).filter(feature=>feature.lowest_endpoint_offset_working_px!==null&&feature.lowest_endpoint_offset_working_px>=-threshold).length;
 const count=row.net_state==='measured'?`${supported} supported posts (bounded ${threshold} px diagnostic)`:`post projection ${row.net_state.replaceAll('_',' ')}${row.projection_reason?` (${row.projection_reason})`:''}`;
 const bonus=setting==='old_fixed'&&name!=='saved'?'old unrestricted bonus —':score?`lower-post bonus ${score.bonus.toFixed(3)}`:'lower-post bonus —';
 return `Full-court rank ${row.full_court_rank} · paint score ${row.paint_score.toFixed(4)} · ${bonus} · ${count} · ${key}`;
}
function visible(data){
 const filter=byId('filter').value;
 if(filter==='all')return true;
 const primary=data.choices.primary!==data.baseline_key;
 const seeded=data.source_label==='am1_seeded_pool';
 const sensitive=Object.values(data.choices).some(key=>key!==data.choices.primary);
 return primary||seeded||data.original_review||(filter==='sensitivity'&&sensitive);
}
function updateVisibility(){
 for(const state of states)state.section.hidden=!visible(state.data);
 for(const link of byId('nav').children)link.hidden=byId(link.getAttribute('href').slice(1)).hidden;
}
for(const [index,data] of cases.entries()){
 const anchor=`case-${index}`;
 const link=document.createElement('a');link.href='#'+anchor;link.textContent=`${data.case_id} (${data.source_label==='am1_seeded_pool'?'seeded Am1':'saved pool'})`;byId('nav').append(link);
 const section=document.createElement('section');section.id=anchor;
 const heading=document.createElement('h2');heading.textContent=data.case_id;section.append(heading);
 const cohort=document.createElement('p');cohort.className='small badge';
 const cohortName=data.source_label==='am1_seeded_pool'?'Seeded Am1 · development':data.cohort==='development'?'Saved pool · development':'Saved pool · additional same-source view';
 cohort.textContent=`${cohortName} · ${data.group} · ${data.scan_label}${data.reference_status?` · source reference status: ${data.reference_status}`:''}${data.view_status?` · ${data.view_status.replaceAll('_',' ')}`:''}`;
 section.append(cohort);
 const summary=document.createElement('p');summary.className='small';section.append(summary);
 const grid=document.createElement('div');grid.className='grid';grid.style.setProperty('--columns',2);section.append(grid);
 const image=new Image(),panels={};
 const baseline=data.views[data.baseline_key];
 const focusCorners=baseline?.geometry.corners||Object.values(data.views)[0]?.geometry.corners;
 const state={data,section,focus:focusCorners?[...focusCorners[0]]:[data.size[0]/2,data.size[1]/2],corner:null};
 for(const name of ['saved','trial']){
  const panel=document.createElement('div');panel.className='panel';
  const title=document.createElement('div');title.className='label';title.style.color=name==='saved'?'#f5f5f0':'#00e5ff';panel.append(title);
  const canvas=document.createElement('canvas');canvas.className='full';canvas.width=data.size[0];canvas.height=data.size[1];panel.append(canvas);
  canvas.addEventListener('click',event=>{const rect=canvas.getBoundingClientRect();state.focus=[(event.clientX-rect.left)/rect.width*data.size[0],(event.clientY-rect.top)/rect.height*data.size[1]];state.corner.value='custom';update();});
  const detail=document.createElement('div');detail.className='label small';panel.append(detail);grid.append(panel);
  panels[name]={title,detail,canvas};
 }
 const cropTitle=document.createElement('div');cropTitle.className='crop-title';
 const focusLabel=document.createElement('label');focusLabel.textContent='Focus ';
 const corner=document.createElement('select');corner.innerHTML=cornerLabels.map((label,i)=>`<option value="${i}">${label}</option>`).join('')+'<option value="custom">Clicked point</option>';
 focusLabel.append(corner);cropTitle.append(focusLabel);const cropNote=document.createElement('span');cropTitle.append(' · ',cropNote);section.append(cropTitle);state.corner=corner;
 const cropGrid=document.createElement('div');cropGrid.className='grid';cropGrid.style.setProperty('--columns',2);section.append(cropGrid);
 for(const name of ['saved','trial']){
  const panel=document.createElement('div');panel.className='panel crop';
  const title=document.createElement('div');title.className='label';title.style.color=name==='saved'?'#f5f5f0':'#00e5ff';panel.append(title);
  const canvas=document.createElement('canvas');canvas.width=600;canvas.height=450;panel.append(canvas);cropGrid.append(panel);
  panels[name].cropTitle=title;panels[name].crop=canvas;
 }
 const note=document.createElement('p');note.className='small';note.textContent='White court: saved paint choice. Cyan court: selected comparison. Dashed amber: projected net diagnostic. Lower-post evidence: amber square marks the projected base; pale circles are uncovered samples; blue squares are covered samples. Solid blue segments are accepted support; dashed amber segments are support vetoed below the projected base. Net projection and lower-post evidence are off by default.';section.append(note);
 byId('gallery').append(section);
 function update(){
  const setting=byId('view').value,key=selectedKey(data,'trial');
  const primaryChanged=data.choices.primary!==data.baseline_key;
  const sensitive=data.choices.weight_02!==data.choices.primary||data.choices.weight_08!==data.choices.primary;
  const changed=key!==data.baseline_key;
  const reviewNote=data.case_id==='shuttleset_21_scene_0034'?' · primary rank 3 is unreviewed (earlier review covered rank 2)':'';
  summary.textContent=`Primary ${primaryChanged?'changed':'unchanged'} · shown choice ${changed?'changed':'unchanged'} · weight-sensitive ${sensitive?'yes':'no'} · score units, not probabilities${reviewNote}`;
  cropNote.textContent=`Enlarged crop · centre (${state.focus[0].toFixed(1)}, ${state.focus[1].toFixed(1)}) · 120 × 90 working pixels`;
  for(const name of ['saved','trial']){
   const panel=panels[name],choice=selectedKey(data,name);
   panel.title.textContent=name==='saved'?'Saved paint choice':choiceLabels[setting];
   panel.detail.textContent=facts(data,choice,name);
   panel.cropTitle.textContent=panel.title.textContent;
   panel.canvas.setAttribute('aria-label',`${data.case_id} ${panel.title.textContent}`);
   panel.crop.setAttribute('aria-label',`${data.case_id} ${panel.title.textContent}, enlarged crop`);
   if(image.complete&&image.naturalWidth){draw(panel.canvas,image,data,name,state.focus,false);draw(panel.crop,image,data,name,state.focus,true);}
  }
 }
 corner.addEventListener('change',()=>{if(corner.value!=='custom')state.focus=focusCorners?[...focusCorners[Number(corner.value)]]:[data.size[0]/2,data.size[1]/2];update();});
 state.reset=()=>{state.focus=focusCorners?[...focusCorners[0]]:[data.size[0]/2,data.size[1]/2];corner.value='0';update();};
 image.onload=update;image.src=data.image;redraw.push(update);states.push(state);update();
}
for(const id of ['view','mode','width','opacity','projected','base','outlines','smooth'])byId(id).addEventListener('input',()=>redraw.forEach(call=>call()));
byId('filter').addEventListener('input',updateVisibility);updateVisibility();
window.addEventListener('resize',()=>redraw.forEach(call=>call()));
byId('reset').addEventListener('click',()=>states.forEach(state=>state.reset()));
document.addEventListener('keydown',event=>{if(event.code==='Space'&&!['INPUT','SELECT','BUTTON'].includes(document.activeElement.tagName)){event.preventDefault();spaceHeld=true;redraw.forEach(call=>call());}});
document.addEventListener('keyup',event=>{if(event.code==='Space'){spaceHeld=false;redraw.forEach(call=>call());}});
window.addEventListener('blur',()=>{spaceHeld=false;redraw.forEach(call=>call());});
