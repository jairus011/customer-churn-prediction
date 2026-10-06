(function(root){
function predict(model, record){
 const missing=model.numeric.map(k=>record[k]===null||record[k]===undefined||record[k]===''||!Number.isFinite(Number(record[k])));
 let values=model.numeric.map((k,i)=>missing[i]?model.medians[i]:Number(record[k]));
 values=values.concat(model.indicators.map(i=>missing[i]?1:0));
 let x=values.map((v,i)=>(v-model.mean[i])/model.scale[i]);
 model.categorical.forEach((k,i)=>model.categories[i].forEach(c=>x.push(record[k]===c?1:0)));
 // sklearn tree inference casts to float32 before traversing.
 x=x.map(Math.fround);
 let score=Math.log(model.prior/(1-model.prior));
 for(const tree of model.trees){let n=0;while(tree.left[n]!==-1)n=x[tree.feature[n]]<=tree.threshold[n]?tree.left[n]:tree.right[n];score+=model.learning_rate*tree.value[n];}
 return 1/(1+Math.exp(-score));
}
if(typeof module!=='undefined')module.exports={predict};else root.churnPredict=predict;
})(typeof window==='undefined'?globalThis:window);
