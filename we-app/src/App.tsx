import { Suspense, lazy } from "react";
import { NavLink, Route, Routes } from "react-router-dom";

const OverviewPage = lazy(() =>
  import("./pages/OverviewPage").then((module) => ({ default: module.OverviewPage })),
);
const FaultModelPage = lazy(() =>
  import("./pages/FaultModelPage").then((module) => ({ default: module.FaultModelPage })),
);

export function App() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Scientific web app</p>
          <h1>L'Aquila Fault Explorer</h1>
          <p className="topbar-subtitle">
            Interactive inspection of the Barisciano-Sulmona fault geometry, GNSS deformation
            histories, and exported PINN model snapshots.
          </p>
        </div>

        <nav className="topbar-nav" aria-label="Primary navigation">
          <NavLink
            to="/"
            end
            className={({ isActive }) => (isActive ? "nav-pill nav-pill-active" : "nav-pill")}
          >
            2D overview
          </NavLink>
          <NavLink
            to="/fault-model"
            className={({ isActive }) => (isActive ? "nav-pill nav-pill-active" : "nav-pill")}
          >
            3D fault model
          </NavLink>
        </nav>
      </header>

      <main className="app-main">
        <Suspense fallback={<div className="route-loading">Loading interface…</div>}>
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/fault-model" element={<FaultModelPage />} />
          </Routes>
        </Suspense>
      </main>
    </div>
  );
}
