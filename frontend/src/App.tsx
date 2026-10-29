import { useEffect, useMemo, useState } from "react";
import {
  ArrowDownRight, ArrowRight, ArrowUpRight, Bell, Boxes, CalendarDays,
  Check, ChevronDown, CircleHelp, Clock3, Command, Download, Filter, Gauge,
  LayoutDashboard, MapPin, Moon, MoreHorizontal, Package, Plus, Search,
  Settings, ShieldCheck, Truck, Users, Wallet, Waypoints,
} from "lucide-react";
import { api } from "./api/endpoints";
import { describeError } from "./api/client";
import type { HealthResponse } from "./api/types";

type ApiState = "checking" | "online" | "offline";

const shipments = [
  { id: "DL-1045", ref: "ORD-88214", from: "Brooklyn depot", to: "Queens Market", eta: "10:40 AM", route: "North loop", status: "In transit", color: "amber" },
  { id: "DL-1044", ref: "ORD-88202", from: "Brooklyn depot", to: "Long Island City", eta: "11:15 AM", route: "East loop", status: "Validated", color: "green" },
  { id: "DL-1046", ref: "ORD-88238", from: "Brooklyn depot", to: "Jersey City", eta: "12:05 PM", route: "West loop", status: "Queued", color: "slate" },
];

const activities = [
  { id: "ORD-88214", customer: "Harbor Market", company: "Northstar Foods", eta: "Today, 10:40 AM", route: "North loop", cost: "$184.20", state: "In transit", color: "amber" },
  { id: "ORD-88202", customer: "Mason & Co.", company: "Mason Retail", eta: "Today, 11:15 AM", route: "East loop", cost: "$96.00", state: "Validated", color: "green" },
  { id: "ORD-88238", customer: "Parkside Grocer", company: "Parkside Group", eta: "Today, 12:05 PM", route: "West loop", cost: "$121.50", state: "Queued", color: "slate" },
  { id: "ORD-88197", customer: "Riverside Cafe", company: "Riverside Foods", eta: "Today, 1:20 PM", route: "South loop", cost: "$78.40", state: "Needs review", color: "rose" },
];

const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

function App() {
  const [apiState, setApiState] = useState<ApiState>("checking");
  const [apiLabel, setApiLabel] = useState("Checking backend");
  const [query, setQuery] = useState("");
  const [period, setPeriod] = useState("This week");
  const [activeNav, setActiveNav] = useState("Overview");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    api.health(controller.signal)
      .then(({ data }: { data: HealthResponse }) => {
        setApiState("online");
        setApiLabel(`API ${data.api_version} connected`);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setApiState("offline");
        const description = describeError(error);
        setApiLabel(description.title === "Backend unreachable" ? "Backend offline" : description.title);
      });
    return () => controller.abort();
  }, []);

  const filteredActivities = useMemo(() => {
    const term = query.trim().toLowerCase();
    return term ? activities.filter((item) => Object.values(item).some((value) => value.toLowerCase().includes(term))) : activities;
  }, [query]);

  function announce(message: string) {
    setNotice(message);
    window.setTimeout(() => setNotice(""), 2600);
  }

  return (
    <div className="app-frame">
      <aside className="sidebar">
        <a className="brand" href="#overview" aria-label="ArQ home"><span className="brand-mark"><Command size={21} strokeWidth={2.5} /></span></a>
        <div className="nav-group">
          {[
            ["Overview", LayoutDashboard], ["Dispatch", Waypoints], ["Scenarios", Boxes],
            ["Fleet", Truck], ["Analytics", Gauge], ["Team", Users],
          ].map(([label, Icon]) => {
            const name = label as string;
            const Glyph = Icon as typeof LayoutDashboard;
            return <button key={name} className={`nav-button ${activeNav === name ? "active" : ""}`} aria-label={name} title={name} onClick={() => { setActiveNav(name); if (name !== "Overview") announce(`${name} workspace is coming next`); }}><Glyph size={19} /><span>{name}</span></button>;
          })}
        </div>
        <div className="sidebar-bottom">
          <button className="nav-button" aria-label="Help" onClick={() => announce("Help center is coming soon")}><CircleHelp size={19} /><span>Help</span></button>
          <button className="nav-button" aria-label="Settings" onClick={() => announce("Settings are coming next")}><Settings size={19} /><span>Settings</span></button>
          <button className="theme-button" aria-label="Toggle appearance" onClick={() => document.body.classList.toggle("dark-mode")}><Moon size={17} /></button>
        </div>
      </aside>

      <main className="main-area" id="overview">
        <header className="topbar">
          <div className="breadcrumb"><span className="breadcrumb-muted">Workspace</span><span className="slash">/</span><strong>{activeNav}</strong></div>
          <div className="topbar-actions">
            <span className={`api-pill ${apiState}`}><i />{apiLabel}</span>
            <button className="icon-button notification" aria-label="Notifications" onClick={() => announce("You’re all caught up")}><Bell size={18} /><b /></button>
            <div className="profile-chip"><div className="avatar">JD</div><div><strong>Jordan Davis</strong><small>Dispatch manager</small></div><ChevronDown size={15} /></div>
          </div>
        </header>

        <div className="content-wrap">
          <section className="welcome-row">
            <div>
              <div className="eyebrow"><span className="eyebrow-dot" /> OPERATIONS OVERVIEW <span className="preview-tag">Preview data</span></div>
              <h1>Good morning, Jordan <span>✳</span></h1>
              <p className="subhead">Here’s your dispatch snapshot. Operational figures below are illustrative until connected to live data.</p>
            </div>
            <div className="welcome-actions">
              <button className="date-button" onClick={() => announce("Date range selector is coming next")}><CalendarDays size={16} /><span>Oct 26 – Nov 01, 2026</span><ChevronDown size={14} /></button>
              <button className="primary-button" onClick={() => announce("Scenario builder is the next workflow to add")}><Plus size={17} /> New scenario</button>
            </div>
          </section>

          <section className="metric-grid" aria-label="Dispatch summary">
            <Metric icon={<Package size={17} />} label="Orders today" value="178" change="12.4%" direction="up" foot="vs. previous week" tone="blue" />
            <Metric icon={<Clock3 size={17} />} label="Awaiting dispatch" value="24" change="4.8%" direction="down" foot="vs. previous week" tone="violet" />
            <Metric icon={<Truck size={17} />} label="Routes planned" value="12" change="2 routes" direction="up" foot="across 9 vehicles" tone="orange" />
            <Metric icon={<ShieldCheck size={17} />} label="Plans validated" value="96.2%" change="3.1%" direction="up" foot="independent validation" tone="green" />
          </section>

          <section className="dashboard-grid">
            <div className="column-main">
              <article className="card map-card">
                <div className="card-heading map-heading">
                  <div><div className="title-with-live"><h2>Dispatch overview</h2><span className="preview-live"><i /> Preview</span></div><p>Route map preview · no live vehicle positions</p></div>
                  <button className="more-button" aria-label="Map options" onClick={() => announce("Map options are coming next")}><MoreHorizontal size={19} /></button>
                </div>
                <div className="map-canvas" role="img" aria-label="Illustrative route map, not live tracking">
                  <div className="map-grid-lines" />
                  <span className="map-label label-one">WILLIAMSBURG</span><span className="map-label label-two">BROOKLYN</span><span className="map-label label-three">EAST RIVER</span><span className="map-label label-four">DOWNTOWN</span>
                  <svg className="route-lines" viewBox="0 0 1000 370" preserveAspectRatio="none" aria-hidden="true">
                    <path d="M-10 265 C130 235 155 248 250 192 S420 155 495 174 S620 245 735 176 S880 88 1010 118" className="route-muted" />
                    <path d="M-10 265 C130 235 155 248 250 192 S420 155 495 174 S620 245 735 176 S880 88 1010 118" className="route-bright" />
                    <path d="M183 340 C250 280 304 265 363 228 S482 178 542 180 S690 205 765 260" className="route-dash" />
                    <path d="M342 34 C375 90 425 128 495 174 S600 276 626 348" className="route-secondary" />
                  </svg>
                  <div className="map-road road-a"/><div className="map-road road-b"/><div className="map-road road-c"/><div className="map-road road-d"/>
                  <div className="map-pin pin-start"><span><Package size={14} /></span><small>Brooklyn hub</small></div>
                  <div className="map-pin pin-stop"><span><MapPin size={15} /></span><small>Queens · 10:40</small></div>
                  <div className="map-pin pin-end"><span><MapPin size={15} /></span><small>Long Island City</small></div>
                  <div className="map-legend"><span><i className="legend-orange" /> Planned route</span><span><i className="legend-gray" /> Alternate</span><span className="map-approx">Schematic preview</span></div>
                  <div className="map-controls"><button aria-label="Zoom in" onClick={() => announce("Map zoom is available when map data is connected")}>+</button><button aria-label="Zoom out" onClick={() => announce("Map zoom is available when map data is connected")}>−</button><button aria-label="Map layers" onClick={() => announce("Map layers are coming next")}><Filter size={15} /></button></div>
                </div>
                <div className="map-footer"><div className="route-footer-title"><span className="vehicle-icon"><Truck size={18} /></span><div><strong>12 planned routes</strong><small>Last plan: awaiting backend connection</small></div></div><button className="text-button" onClick={() => { setActiveNav("Dispatch"); announce("Dispatch workspace is coming next"); }}>View dispatch <ArrowRight size={15} /></button></div>
              </article>

              <article className="card table-card">
                <div className="card-heading table-heading">
                  <div><h2>Recent activity</h2><p>Preview rows · replace with validated backend runs</p></div>
                  <div className="table-tools"><div className="search-box"><Search size={16} /><input aria-label="Search activity" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search orders" /></div><button className="icon-button filter-button" aria-label="Filter activity" onClick={() => announce("Activity filters are coming next")}><Filter size={16} /></button><button className="export-button" onClick={() => { const csv = ["Order,Customer,Company,ETA,Route,Cost,Status", ...filteredActivities.map((r) => [r.id,r.customer,r.company,r.eta,r.route,r.cost,r.state].join(","))].join("\n"); const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" })); link.download = "arq-preview-activity.csv"; link.click(); URL.revokeObjectURL(link.href); }}><Download size={15} /><span>Export</span></button></div>
                </div>
                <div className="table-scroll"><table><thead><tr><th>Order</th><th>Customer</th><th>Company</th><th>ETA</th><th>Route</th><th>Est. cost</th><th>Status</th><th /></tr></thead><tbody>
                  {filteredActivities.map((row) => <tr key={row.id}><td><strong className="order-id">{row.id}</strong></td><td><div className="customer-cell"><span className="customer-avatar">{row.customer.split(" ").map((x) => x[0]).join("").slice(0,2)}</span><span>{row.customer}</span></div></td><td className="company-cell">{row.company}</td><td>{row.eta}</td><td>{row.route}</td><td className="cost-cell">{row.cost}</td><td><span className={`status-badge ${row.color}`}><i />{row.state}</span></td><td><button className="row-more" aria-label={`More actions for ${row.id}`} onClick={() => announce(`Actions for ${row.id} are coming next`)}><MoreHorizontal size={17} /></button></td></tr>)}
                  {filteredActivities.length === 0 && <tr><td colSpan={8} className="empty-row">No preview activity matches “{query}”.</td></tr>}
                </tbody></table></div>
                <div className="table-bottom"><span>Showing {filteredActivities.length} of 4 preview rows</span><button className="text-button" onClick={() => announce("Run history is coming next")}>Open run history <ArrowRight size={15} /></button></div>
              </article>
            </div>

            <aside className="column-side">
              <article className="card tracking-card">
                <div className="card-heading"><div><div className="section-kicker">ROUTE SNAPSHOT</div><h2>Delivery tracking</h2></div><button className="more-button" aria-label="Tracking options" onClick={() => announce("Tracking options are coming next")}><MoreHorizontal size={19} /></button></div>
                <div className="tracking-search"><Search size={15} /><span>DL-1045</span><button onClick={() => announce("Tracking lookup will use backend job and route data")}>Lookup</button></div>
                <div className="shipment-route">
                  {shipments.map((item) => <div className="shipment-item" key={item.id}><div className="timeline-dot-wrap"><span className={`timeline-dot ${item.color}`} /><span className="timeline-stem" /></div><div className="shipment-copy"><div className="shipment-top"><strong>{item.from}</strong><small>{item.eta}</small></div><div className="shipment-bottom"><span>{item.id} · {item.route}</span><span className={`mini-status ${item.color}`}>{item.status}</span></div></div></div>)}
                </div>
                <button className="outline-wide" onClick={() => announce("Live tracking requires a driver location feed")}>Open route details <ArrowRight size={15} /></button>
              </article>

              <article className="card efficiency-card">
                <div className="card-heading"><div><div className="section-kicker">PERFORMANCE</div><h2>Route efficiency</h2></div><button className="mini-select" onClick={() => setPeriod(period === "This week" ? "Last week" : "This week")}>{period}<ChevronDown size={12} /></button></div>
                <div className="efficiency-value"><strong>—</strong><span>Needs connected run data</span></div>
                <div className="line-chart"><svg viewBox="0 0 340 100" preserveAspectRatio="none" aria-label="Illustrative weekly trend"><defs><linearGradient id="area" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#ffad55" stopOpacity=".22"/><stop offset="1" stopColor="#ffad55" stopOpacity="0"/></linearGradient></defs><path d="M0 75 C28 61 36 37 62 48 S104 76 132 55 S172 31 198 47 S234 65 260 40 S300 31 340 15 L340 100 L0 100Z" fill="url(#area)"/><path d="M0 75 C28 61 36 37 62 48 S104 76 132 55 S172 31 198 47 S234 65 260 40 S300 31 340 15" fill="none" stroke="#f39b3d" strokeWidth="3" strokeLinecap="round"/><path d="M0 87 C33 75 47 67 77 76 S116 86 144 71 S176 65 204 74 S250 88 274 66 S311 70 340 50" fill="none" stroke="#b7bfca" strokeWidth="2" strokeLinecap="round"/></svg></div>
                <div className="chart-legend"><span><i className="orange-square" />This period</span><span><i className="gray-square" />Previous</span></div>
                <div className="chart-periods">{days.map((d) => <span key={d}>{d}</span>)}</div>
                <div className="preview-note">Trend graphic is illustrative; no efficiency metric is currently returned by the API.</div>
              </article>

              <article className="card cost-card">
                <div className="card-heading"><div><div className="section-kicker">FINANCIAL VIEW</div><h2>Plan cost summary</h2></div><Wallet size={18} className="muted-icon" /></div>
                <div className="cost-list"><div><span>Selected plan</span><strong>—</strong></div><div><span>Travel minutes</span><strong>—</strong></div><div><span>SLA exposure</span><strong>—</strong></div></div>
                <div className="cost-callout"><span className="callout-icon"><Check size={15} /></span><span>Costs appear here after a plan is calculated and independently validated.</span></div>
              </article>

              <article className="dispatch-tip"><div className="tip-icon"><Waypoints size={18} /></div><div><strong>Build a decision you can verify</strong><p>Start with a scenario, run a plan, then review its independent validation.</p><button onClick={() => announce("Scenario builder is the next workflow to add")}>Start a scenario <ArrowRight size={14} /></button></div></article>
            </aside>
          </section>

          <footer className="page-footer"><span>ArQ Dispatch · Local prototype</span><span>Operational values require backend data</span></footer>
        </div>
      </main>
      {notice && <div className="toast" role="status">{notice}</div>}
    </div>
  );
}

function Metric({ icon, label, value, change, direction, foot, tone }: { icon: React.ReactNode; label: string; value: string; change: string; direction: "up" | "down"; foot: string; tone: string }) {
  return <article className="metric-card"><div className="metric-top"><span className={`metric-icon ${tone}`}>{icon}</span><button className="metric-menu" aria-label={`${label} info`}><MoreHorizontal size={17} /></button></div><div className="metric-label">{label}</div><div className="metric-value-row"><strong>{value}</strong><span className={`metric-change ${direction}`} >{direction === "up" ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}{change}</span></div><div className="metric-foot">{foot}</div></article>;
}

export default App;
