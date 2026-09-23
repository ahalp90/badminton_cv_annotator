'use strict';
const cases=__GALLERY_DATA__;
const cornerLabels=['Upper left','Upper right','Lower right','Lower left'];
const cropWidth=120,cropHeight=90;
const states=[],redraw=[];
let spaceHeld=false;
const byId=id=>document.getElementById(id);
function outline(corners){return corners.map((point,index)=>[point,corners[(index+1)%4]]);}
function strokeLines(ctx,lines,colour,thickness,scale){
 ctx.save();ctx.beginPath();
 for(const [start,end] of lines){ctx.moveTo(...start);ctx.lineTo(...end);}
 ctx.strokeStyle='#071019';ctx.lineWidth=(thickness+3)*scale;ctx.stroke();
 ctx.strokeStyle=colour;ctx.lineWidth=thickness*scale;ctx.stroke();ctx.restore();
}
function draw(canvas,image,data,name,focus,isCrop){
 const ctx=canvas.getContext('2d');
 const box=isCrop?[focus[0]-cropWidth/2,focus[1]-cropHeight/2,cropWidth,cropHeight]:[0,0,...data.size];
 const sx=canvas.width/box[2],sy=canvas.height/box[3];
 ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle='#050b10';ctx.fillRect(0,0,canvas.width,canvas.height);
 ctx.imageSmoothingEnabled=!isCrop||byId('smooth').checked;
 ctx.setTransform(sx,0,0,sy,-box[0]*sx,-box[1]*sy);ctx.drawImage(image,0,0,...data.size);
 if(!byId('outlines').checked||spaceHeld)return;
 const thickness=Number(byId('width').value),scale=box[2]/canvas.getBoundingClientRect().width;
 ctx.globalAlpha=Number(byId('opacity').value);ctx.lineJoin='round';
 const geometry=data[name],mode=byId('mode').value;
 const lines=mode==='edges'?geometry.edges: outline(geometry.corners);
 strokeLines(ctx,lines,name==='before'?'#f5f5f0':'#00e5ff',thickness,scale);
 if(mode==='centres')strokeLines(ctx,geometry.centres,'#ffb547',Math.max(1.5,thickness*.7),scale);
 ctx.globalAlpha=1;
}
function textFor(data){
 const fit=data.fit_status.replaceAll('_',' ');
 return `${data.changed_fragments}/${data.fragment_count} fragments changed · ${data.unresolved_fragments} unresolved · fit ${fit} (${data.fit_valid?'valid':'invalid'}) · camera gate ${data.camera_eligible?'eligible':'ineligible'} · maximum corner movement ${data.maximum_corner_movement_working_px.toFixed(2)} working px (diagnostic) · original replay ${data.original_replay_matched?'matched':'unmatched'}`;
}
for(const [index,data] of cases.entries()){
 const anchor=`case-${index}`;
 const link=document.createElement('a');link.href='#'+anchor;
 link.textContent=`${data.case_id} (${data.source_label==='am1_seeded_pool'?'seeded Am1':'saved pool'})`;
 byId('nav').append(link);
 const section=document.createElement('section');section.id=anchor;
 const heading=document.createElement('h2');heading.textContent=data.case_id;section.append(heading);
 const cohort=document.createElement('p');cohort.className='small badge';
 cohort.textContent=`${data.source_label==='am1_seeded_pool'?'Seeded Am1':'Saved pool'} · ${data.cohort}`;
 section.append(cohort);
 const summary=document.createElement('p');summary.className='small';summary.textContent=textFor(data);section.append(summary);
 const grid=document.createElement('div');grid.className='grid';grid.style.setProperty('--columns',2);section.append(grid);
 const image=new Image(),panels={};
 const focusCorners=data.before.corners;
 const state={focus:[...focusCorners[0]],corner:null};
 for(const name of ['before','after']){
  const panel=document.createElement('div');panel.className='panel';
  const title=document.createElement('div');title.className='label';
  title.textContent=name==='before'?'Selected court before correction':'Automatic stripe-polarity correction';
  title.style.color=name==='before'?'#f5f5f0':'#00e5ff';panel.append(title);
  const canvas=document.createElement('canvas');canvas.className='full';canvas.width=data.size[0];canvas.height=data.size[1];panel.append(canvas);
  canvas.setAttribute('aria-label',`${data.case_id} ${title.textContent}`);
  canvas.addEventListener('click',event=>{
   const rect=canvas.getBoundingClientRect();
   state.focus=[(event.clientX-rect.left)/rect.width*data.size[0],(event.clientY-rect.top)/rect.height*data.size[1]];
   state.corner.value='custom';update();
  });
  const detail=document.createElement('div');detail.className='label small';
  detail.textContent=name==='before'?'Current bounded primary selection':'Saved corrected fit';panel.append(detail);grid.append(panel);
  panels[name]={title,canvas};
 }
 const cropTitle=document.createElement('div');cropTitle.className='crop-title';
 const focusLabel=document.createElement('label');focusLabel.textContent='Focus ';
 const corner=document.createElement('select');
 corner.innerHTML=cornerLabels.map((label,cornerIndex)=>`<option value="${cornerIndex}">${label}</option>`).join('')+
  '<option value="custom">Clicked or dragged point</option>';
 focusLabel.append(corner);cropTitle.append(focusLabel);
 const cropNote=document.createElement('span');cropTitle.append(' · ',cropNote);section.append(cropTitle);state.corner=corner;
 const cropGrid=document.createElement('div');cropGrid.className='grid';cropGrid.style.setProperty('--columns',2);section.append(cropGrid);
 for(const name of ['before','after']){
  const panel=document.createElement('div');panel.className='panel crop';
  const title=document.createElement('div');title.className='label';title.textContent=panels[name].title.textContent;
  title.style.color=name==='before'?'#f5f5f0':'#00e5ff';panel.append(title);
  const canvas=document.createElement('canvas');canvas.width=600;canvas.height=450;panel.append(canvas);cropGrid.append(panel);
  canvas.setAttribute('aria-label',`${data.case_id} ${title.textContent}, enlarged crop`);
  canvas.style.cursor='grab';
  let pointer=null;
  canvas.addEventListener('pointerdown',event=>{pointer=[event.clientX,event.clientY,...state.focus];canvas.setPointerCapture(event.pointerId);canvas.style.cursor='grabbing';});
  canvas.addEventListener('pointermove',event=>{
   if(!pointer)return;
   const rect=canvas.getBoundingClientRect();
   state.focus=[pointer[2]-(event.clientX-pointer[0])/rect.width*cropWidth,
                pointer[3]-(event.clientY-pointer[1])/rect.height*cropHeight];
   corner.value='custom';update();
  });
  canvas.addEventListener('pointerup',()=>{pointer=null;canvas.style.cursor='grab';});
  canvas.addEventListener('pointercancel',()=>{pointer=null;canvas.style.cursor='grab';});
  panels[name].crop=canvas;
 }
 const note=document.createElement('p');note.className='small';
 note.textContent='White: bounded primary court. Cyan: corrected fit. Orange: stripe centres when selected. Overlay modes show the court boundary, stripe edges or stripe centres. Net and post projections were not recalculated for these fits.';
 section.append(note);byId('gallery').append(section);
 function update(){
  cropNote.textContent=`Enlarged crop · centre (${state.focus[0].toFixed(1)}, ${state.focus[1].toFixed(1)}) · 120 × 90 working pixels`;
  if(image.complete&&image.naturalWidth){
   for(const name of ['before','after']){
    draw(panels[name].canvas,image,data,name,state.focus,false);
    draw(panels[name].crop,image,data,name,state.focus,true);
   }
  }
 }
 corner.addEventListener('change',()=>{if(corner.value!=='custom')state.focus=[...focusCorners[Number(corner.value)]];update();});
 state.reset=()=>{state.focus=[...focusCorners[0]];corner.value='0';update();};
 image.onload=update;image.src=data.image;redraw.push(update);states.push(state);update();
}
for(const id of ['mode','width','opacity','outlines','smooth'])byId(id).addEventListener('input',()=>redraw.forEach(call=>call()));
window.addEventListener('resize',()=>redraw.forEach(call=>call()));
byId('reset').addEventListener('click',()=>states.forEach(state=>state.reset()));
document.addEventListener('keydown',event=>{if(event.code==='Space'&&!['INPUT','SELECT','BUTTON'].includes(document.activeElement.tagName)){event.preventDefault();spaceHeld=true;redraw.forEach(call=>call());}});
document.addEventListener('keyup',event=>{if(event.code==='Space'){spaceHeld=false;redraw.forEach(call=>call());}});
window.addEventListener('blur',()=>{spaceHeld=false;redraw.forEach(call=>call());});
