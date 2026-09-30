"use client";

import { useEffect, useMemo, useState } from "react";
import { ArrowUpRight, BrainCircuit, ChevronRight, CircleGauge, Crosshair, RefreshCw, Search, ShieldAlert, Sparkles, UsersRound, X } from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

type Customer = {customer_id:string; churn_probability:number; risk:string; membership:string; region:string; orders_90d:number; days_since_last_purchase:number; lifetime_value:number; retention_action:string};
type Detail = {profile:Record<string,any>; behavior:Record<string,any>; prediction:{probability:number;risk:string}; drivers:{feature:string;impact:number;direction:string}[]; retention_action:string};

function money(v:number){return `$${Math.round(v).toLocaleString()}`}
function pct(v:number){return `${(v*100).toFixed(1)}%`}

export default function Home(){
  const [summary,setSummary]=useState<any>(null); const [customers,setCustomers]=useState<Customer[]>([]); const [segments,setSegments]=useState<any[]>([]); const [selected,setSelected]=useState<Detail|null>(null); const [query,setQuery]=useState(""); const [risk,setRisk]=useState("all"); const [loading,setLoading]=useState(true);
  const load=async()=>{setLoading(true); try{const [s,c,g]=await Promise.all([fetch(`${API}/api/summary`).then(r=>r.json()),fetch(`${API}/api/customers?limit=120`).then(r=>r.json()),fetch(`${API}/api/segments`).then(r=>r.json())]); setSummary(s); setCustomers(c); setSegments(g)}finally{setLoading(false)}};
  useEffect(()=>{load()},[]);
  const filtered=useMemo(()=>customers.filter(c=>(risk==="all"||c.risk===risk)&&(`${c.customer_id} ${c.membership} ${c.region}`).toLowerCase().includes(query.toLowerCase())),[customers,query,risk]);
  const bars=(summary?.recent_recency_curve||[]).map((x:any)=>Math.max(18, x.churn_rate*210));
  const open=async(id:string)=>setSelected(await fetch(`${API}/api/customers/${id}`).then(r=>r.json()));
  return <div className="app"><div className="shell">
    <header className="topbar"><div className="brand"><div className="brandMark">RQ</div><div><h1>RetentionIQ</h1><span>Customer churn intelligence</span></div></div><div className="status"><i className="dot"/> ML service live <span>•</span> Random Forest</div></header>
    <section className="hero"><div><div className="eyebrow">E-COMMERCE ML CONTROL CENTER</div><h2>Turn churn probability into retention action.</h2><p>Model customer behavior, identify revenue at risk, understand why customers are likely to churn, and route them into targeted retention plays.</p></div><div className="heroActions"><button className="btn" onClick={load}><RefreshCw size={15}/> Refresh</button></div></section>
    <section className="grid4">
      <div className="kpi"><div className="label">Customers scored</div><div className="value">{summary?summary.customers.toLocaleString():"—"}</div><div className="delta">Live model scoring</div></div>
      <div className="kpi"><div className="label">Baseline churn rate</div><div className="value">{summary?pct(summary.churn_rate):"—"}</div><div className="delta">Observed demo cohort</div></div>
      <div className="kpi"><div className="label">High-risk customers</div><div className="value">{summary?summary.high_risk_customers.toLocaleString():"—"}</div><div className="delta">Priority retention pool</div></div>
      <div className="kpi"><div className="label">Revenue at risk</div><div className="value">{summary?money(summary.retention_value_at_risk):"—"}</div><div className="delta">LTV of high-risk pool</div></div>
    </section>
    <section className="mainGrid">
      <div className="panel"><div className="panelHead"><div><h3>Churn by recency window</h3><span>Observed cohort relationship · days since last purchase</span></div><CircleGauge size={18} color="var(--accent2)"/></div><div className="chart">{bars.map((h:number,i:number)=><div key={i} className="bar" style={{height:`${Math.min(h,200)}px`}}/> )}</div><div className="chartLabels">{(summary?.recent_recency_curve||[]).map((x:any)=><span key={x.bucket}>{x.bucket.replace("["," ").replace("]","")}</span>)}</div></div>
      <div className="panel"><div className="panelHead"><div><h3>Risk distribution</h3><span>Current scored population</span></div><ShieldAlert size={18} color="var(--warn)"/></div><div className="riskList">{["Critical","High","Watch","Healthy"].map(k=><div className="riskRow" key={k}><div><b>{k}</b><small>{summary?.risk_bands?.[k]||0} customers</small></div><span className={`badge ${k}`}>{summary?.risk_bands?.[k]?pct((summary.risk_bands[k]/summary.customers)):"0.0%"}</span></div>)}</div></div>
    </section>
    <section className="two">
      <div className="panel"><div className="panelHead"><div><h3>Customer segments</h3><span>Average modeled risk by membership tier</span></div><UsersRound size={18}/></div>{segments.map((s:any)=><div className="segment" key={s.segment}><div className="segmentTop"><b>{s.segment}</b><span>{pct(s.avg_churn_probability)}</span></div><div className="meter"><i style={{width:`${Math.min(100,s.avg_churn_probability*100)}%`}}/></div><div className="sub" style={{marginTop:8}}>{s.customers.toLocaleString()} customers · Avg LTV {money(s.avg_ltv)}</div></div>)}</div>
      <div className="panel"><div className="panelHead"><div><h3>ML workflow</h3><span>From behavior to action</span></div><BrainCircuit size={18} color="var(--accent)"/></div>{[["Behavior signals","Recency, frequency, engagement, support"],["Model scoring","Logistic Regression + Random Forest"],["Explainability","Local feature contributions"],["Retention action","Segmented save / nurture play"]].map(([a,b],i)=><div className="driver" key={a}><span><b>{i+1}. {a}</b><br/><span className="sub">{b}</span></span><ChevronRight size={15}/></div>)}</div>
    </section>
    <section className="panel tablePanel"><div className="panelHead"><div><h3>Priority customer queue</h3><span>Highest predicted churn probability first</span></div><div className="searchbar"><div style={{position:"relative"}}><Search size={15} style={{position:"absolute",left:10,top:11,color:"var(--muted)"}}/><input className="search" style={{paddingLeft:32}} placeholder="Search customer, tier, region" value={query} onChange={e=>setQuery(e.target.value)}/></div><select className="search" value={risk} onChange={e=>setRisk(e.target.value)}><option value="all">All risk</option><option>Critical</option><option>High</option><option>Watch</option><option>Healthy</option></select></div></div>
      <div className="tableWrap"><table className="table"><thead><tr><th>Customer</th><th>Risk</th><th>Churn probability</th><th>Recency</th><th>Orders</th><th>LTV</th><th>Suggested action</th><th></th></tr></thead><tbody>{filtered.slice(0,40).map(c=><tr key={c.customer_id}><td><div className="customer">{c.customer_id}</div><div className="sub">{c.membership} · {c.region}</div></td><td><span className={`badge ${c.risk}`}>{c.risk}</span></td><td><span className="prob">{pct(c.churn_probability)}</span></td><td>{c.days_since_last_purchase}d</td><td>{c.orders_90d}</td><td>{money(c.lifetime_value)}</td><td className="sub">{c.retention_action}</td><td><button className="btn" onClick={()=>open(c.customer_id)} style={{padding:"7px 9px"}}><ArrowUpRight size={14}/></button></td></tr>)}</tbody></table></div>
    </section>
    <div className="footerNote">Demo data is synthetic and intended for ML/product demonstration. Churn probability is a model output, not a guarantee of future customer behavior.</div>
  </div>
  {selected && <div className="drawerBackdrop" onClick={()=>setSelected(null)}><aside className="drawer" onClick={e=>e.stopPropagation()}><div className="drawerHead"><div><div className="eyebrow">CUSTOMER PROFILE</div><h2 style={{fontFamily:"Space Grotesk",margin:"0 0 5px",fontSize:27}}>{selected.profile.customer_id}</h2><div className="sub">{selected.profile.membership} · {selected.profile.region} · {selected.profile.acquisition_channel}</div></div><button className="close" onClick={()=>setSelected(null)}><X/></button></div><div className="heroRisk"><div className="sub">Predicted churn probability</div><div className="riskNumber">{pct(selected.prediction.probability)}</div><span className={`badge ${selected.prediction.risk}`}>{selected.prediction.risk} risk</span></div><div className="panelHead"><h3>Why the model is concerned</h3><Sparkles size={16} color="var(--accent)"/></div>{selected.drivers.map((d:any)=><div className="driver" key={d.feature}><span><b>{d.feature}</b></span><em className={d.impact>0?"positive":"negative"}>{d.impact>0?"↑":"↓"} {d.direction}</em></div>)}<div className="action"><b>Retention play</b><div className="sub" style={{marginTop:5}}>{selected.retention_action}</div></div><div className="footerNote">Use model explanations as decision support. Validate campaigns and eligibility rules with your business policies.</div></aside></div>}
  </div>
}
