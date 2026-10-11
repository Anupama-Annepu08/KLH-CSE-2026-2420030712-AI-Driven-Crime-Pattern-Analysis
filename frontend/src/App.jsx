import { Fragment, useEffect, useMemo, useState } from "react";
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Popup,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

const HOTSPOTS = [
  { beat: "1834.0", crimes: 598, lat: 41.89236, lon: -87.624138 },
  { beat: "421.0", crimes: 396, lat: 41.755781, lon: -87.561405 },
  { beat: "632.0", crimes: 382, lat: 41.737751, lon: -87.604911 },
  { beat: "123.0", crimes: 380, lat: 41.871362, lon: -87.627619 },
  { beat: "1831.0", crimes: 355, lat: 41.890057, lon: -87.632551 },
  { beat: "1112.0", crimes: 347, lat: 41.898045, lon: -87.720064 },
  { beat: "823.0", crimes: 345, lat: 41.779034, lon: -87.703122 },
  { beat: "511.0", crimes: 342, lat: 41.713253, lon: -87.618906 },
  { beat: "2533.0", crimes: 341, lat: 41.909636, lon: -87.745918 },
  { beat: "111.0", crimes: 340, lat: 41.88465, lon: -87.627877 },
];

const navigation = [
  { name: "Overview", icon: "⌂" },
  { name: "Risk Map", icon: "◈" },
  { name: "Crime Analysis", icon: "◌" },
  { name: "Hotspots", icon: "⌁" },
  { name: "Temporal Analysis", icon: "◷" },
  { name: "Evidence", icon: "◇" },
  { name: "Datasets", icon: "▦" },
  { name: "Research", icon: "▤" },
];

function FlyToLocation({ location }) {
  const map = useMap();

  useEffect(() => {
    if (!location) return;

    map.flyTo(
      [location.lat, location.lon],
      12.1,
      {
        animate: true,
        duration: 2.2,
        easeLinearity: 0.18,
      }
    );
  }, [location, map]);

  return null;
}

function InvestigationPosition({ location, onPositionChange }) {
  const map = useMap();

  useEffect(() => {
    if (!location) {
      onPositionChange(null);
      return undefined;
    }

    const updatePosition = () => {
      const point = map.latLngToContainerPoint([
        location.lat,
        location.lon,
      ]);

      const size = map.getSize();

      const cardWidth = 330;
      const cardHeight = 290;
      const cardGap = 42;

      let cardLeft = point.x + cardGap;
      let cardTop = point.y - 115;

      if (cardLeft + cardWidth > size.x - 18) {
        cardLeft = point.x - cardWidth - cardGap;
      }

      cardLeft = Math.max(
        18,
        Math.min(cardLeft, size.x - cardWidth - 18)
      );

      cardTop = Math.max(
        18,
        Math.min(cardTop, size.y - cardHeight - 18)
      );

      onPositionChange({
        anchorLeft: point.x,
        anchorTop: point.y,
        cardLeft,
        cardTop,
      });
    };

    updatePosition();

    map.on("move", updatePosition);
    map.on("zoom", updatePosition);
    map.on("resize", updatePosition);

    return () => {
      map.off("move", updatePosition);
      map.off("zoom", updatePosition);
      map.off("resize", updatePosition);
    };
  }, [location, map, onPositionChange]);

  return null;
}


const pagePanelStyle = {
  padding: "34px",
  minHeight: "calc(100vh - 110px)",
  color: "#e8e5df",
  background: "radial-gradient(circle at 85% 0%, rgba(169,103,75,.12), transparent 35%), #292a27",
};
const metricGridStyle = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
  gap: "14px",
  margin: "28px 0",
};
const infoCardStyle = {
  padding: "26px",
  border: "1px solid #4c4a43",
  borderRadius: "12px",
  background: "#30312d",
  lineHeight: 1.7,
  maxWidth: "1000px",
};
function PageHeading({ eyebrow, title, subtitle }) {
  return <div style={{ marginBottom: "20px" }}>
    <div style={{ color: "#a6aa9f", fontSize: "10px", letterSpacing: "2px", marginBottom: "12px" }}>{eyebrow}</div>
    <h2 style={{ fontSize: "32px", margin: "0 0 10px" }}>{title}</h2>
    <p style={{ color: "#b6b2a9", maxWidth: "760px", lineHeight: 1.6 }}>{subtitle}</p>
  </div>;
}
function Metric({ label, value }) {
  return <div style={{ padding: "22px", border: "1px solid #4c4a43", borderRadius: "10px", background: "#30312d" }}>
    <div style={{ color: "#a6aa9f", fontSize: "10px", letterSpacing: "1.4px", marginBottom: "12px" }}>{label.toUpperCase()}</div>
    <div style={{ fontSize: "21px", fontWeight: 650 }}>{value}</div>
  </div>;
}

function App() {
  const [activePage, setActivePage] = useState("Overview");
  const [selected, setSelected] = useState(null);
  const [investigating, setInvestigating] = useState(false);
  const [investigationPosition, setInvestigationPosition] =
    useState(null);

  const [prediction, setPrediction] = useState(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState("");
  const [apiStatus, setApiStatus] = useState("checking");
  const [healthData, setHealthData] = useState(null);
  const [datasetSummary, setDatasetSummary] = useState(null);
  const [crimeAnalysis, setCrimeAnalysis] = useState(null);
  const [temporalData, setTemporalData] = useState([]);
  const [temporalGranularity, setTemporalGranularity] = useState("weekly");
  const [lastSync, setLastSync] = useState(null);
  const [dataError, setDataError] = useState("");

  useEffect(() => {
    let cancelled = false;
    let timer;

    const refreshLiveData = async () => {
      try {
        const [healthResponse, summaryResponse, temporalResponse, analysisResponse] = await Promise.all([
          fetch(`${API_BASE}/health`),
          fetch(`${API_BASE}/chicago/summary`),
          fetch(`${API_BASE}/chicago/temporal?granularity=${temporalGranularity}`),
          fetch(`${API_BASE}/chicago/analysis`),
        ]);

        if (!healthResponse.ok || !summaryResponse.ok || !temporalResponse.ok || !analysisResponse.ok) {
          throw new Error("One or more GARUDA API endpoints returned an error.");
        }

        const [health, summary, temporal, analysis] = await Promise.all([
          healthResponse.json(),
          summaryResponse.json(),
          temporalResponse.json(),
          analysisResponse.json(),
        ]);

        if (cancelled) return;
        setApiStatus(health.status === "healthy" ? "online" : "degraded");
        setHealthData(health);
        setDatasetSummary(summary);
        setCrimeAnalysis(analysis);
        setTemporalData(Array.isArray(temporal.records) ? temporal.records : (Array.isArray(temporal.series) ? temporal.series : (Array.isArray(temporal.data) ? temporal.data : [])));
        setLastSync(new Date());
        setDataError("");
      } catch (error) {
        if (cancelled) return;
        setApiStatus("offline");
        setDataError(error?.message || "Unable to reach the GARUDA API.");
      }
    };

    refreshLiveData();
    timer = window.setInterval(refreshLiveData, 30000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [temporalGranularity]);

  const selectedPosition = useMemo(
    () =>
      selected
        ? {
            lat: selected.lat,
            lon: selected.lon,
          }
        : null,
    [selected]
  );

  const handlePositionChange = useMemo(
    () => (position) => {
      setInvestigationPosition(position);
    },
    []
  );

  /*
   * Build the six inputs required by /predict
   * from the observed Chicago records for the
   * selected beat.
   */
  const buildPredictionInput = async (spot) => {
    const response = await fetch(
      `${API_BASE}/chicago/crimes?limit=5000&beat=${encodeURIComponent(
        spot.beat
      )}`
    );

    if (!response.ok) {
      throw new Error(
        `Crime data request failed: ${response.status}`
      );
    }

    const data = await response.json();

    if (
      data.status !== "success" ||
      !Array.isArray(data.records)
    ) {
      throw new Error("Invalid Chicago crime response.");
    }

    const records = data.records;

    const dates = records
      .map((record) => new Date(record.date))
      .filter((date) => !Number.isNaN(date.getTime()));

    if (!dates.length) {
      throw new Error(
        "No valid crime dates were returned for this beat."
      );
    }

    /*
     * The backend's crime_count_7d and crime_count_30d
     * are expected to represent recent observed activity.
     *
     * We calculate them from the most recent observed
     * date available for this beat.
     */
    const latestTimestamp = Math.max(
      ...dates.map((date) => date.getTime())
    );

    const latestDate = new Date(latestTimestamp);

    const sevenDaysAgo = new Date(latestDate);
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);

    const thirtyDaysAgo = new Date(latestDate);
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

    const sameDayRecords = records.filter((record) => {
      const date = new Date(record.date);

      return (
        !Number.isNaN(date.getTime()) &&
        date.toDateString() === latestDate.toDateString()
      );
    });

    const sevenDayRecords = records.filter((record) => {
      const date = new Date(record.date);

      return (
        !Number.isNaN(date.getTime()) &&
        date >= sevenDaysAgo &&
        date <= latestDate
      );
    });

    const thirtyDayRecords = records.filter((record) => {
      const date = new Date(record.date);

      return (
        !Number.isNaN(date.getTime()) &&
        date >= thirtyDaysAgo &&
        date <= latestDate
      );
    });

    const violentTypes = new Set([
      "HOMICIDE",
      "ASSAULT",
      "BATTERY",
      "ROBBERY",
      "CRIM SEXUAL ASSAULT",
      "KIDNAPPING",
      "ARSON",
      "INTIMIDATION",
      "STALKING",
      "OFFENSE INVOLVING CHILDREN",
    ]);

    const violentCrimeCount = sameDayRecords.filter(
      (record) =>
        violentTypes.has(
          String(record.primary_type || "")
            .trim()
            .toUpperCase()
        )
    ).length;

    return {
      daily_crime_count: sameDayRecords.length,
      violent_crime_count: violentCrimeCount,
      crime_count_7d: sevenDayRecords.length,
      crime_count_30d: thirtyDayRecords.length,
      latitude: spot.lat,
      longitude: spot.lon,
    };
  };

  const requestPrediction = async (spot) => {
    setPrediction(null);
    setPredictionError("");
    setPredictionLoading(true);

    try {
      const input = await buildPredictionInput(spot);

      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(input),
      });

      if (!response.ok) {
        throw new Error(
          `Prediction request failed: ${response.status}`
        );
      }

      const result = await response.json();

      if (result.status !== "prediction_successful") {
        throw new Error(
          result.message || "Prediction failed."
        );
      }

      setPrediction({
        input,
        result,
      });
    } catch (error) {
      console.error("GARUDA prediction error:", error);

      setPredictionError(
        error.message ||
          "Unable to retrieve GARUDA prediction."
      );
    } finally {
      setPredictionLoading(false);
    }
  };

  const handleSelect = (spot) => {
    setInvestigating(false);
    setInvestigationPosition(null);
    setPrediction(null);
    setPredictionError("");
    setSelected(spot);
  };

  const handleInvestigate = async () => {
    if (!selected) return;

    setInvestigating(true);

    await requestPrediction(selected);
  };

  const handleClose = () => {
    setSelected(null);
    setInvestigating(false);
    setInvestigationPosition(null);
    setPrediction(null);
    setPredictionError("");
  };

  const getSignalPercent = (value) => {
    if (value === undefined || value === null) {
      return 0;
    }

    return Math.round(Number(value) * 100);
  };

  const getAnomalyPercent = () => {
    if (!prediction) return 0;

    const signal =
      prediction.result?.signals?.anomaly_signal;

    return signal ? 100 : 0;
  };

  const getSpatialPercent = () => {
    if (!prediction) return 0;

    const signal =
      prediction.result?.signals?.spatial_signal;

    return signal ? 100 : 0;
  };

  return (
    <div className="garuda-app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">G</div>

          <div>
            <h1>GARUDA</h1>
            <span>Crime Intelligence</span>
          </div>
        </div>

        <div className="sidebar-section">
          <p className="section-label">
            INTELLIGENCE
          </p>

          <nav>
            {navigation.slice(0, 6).map((item) => (
              <button
                key={item.name}
                className={`nav-item ${
                  activePage === item.name ? "active" : ""
                }`}
                onClick={() =>
                  setActivePage(item.name)
                }
              >
                <span className="nav-icon">
                  {item.icon}
                </span>

                <span>{item.name}</span>
              </button>
            ))}
          </nav>
        </div>

        <div className="sidebar-section secondary">
          <p className="section-label">SYSTEM</p>

          <nav>
            {navigation.slice(6).map((item) => (
              <button
                key={item.name}
                className={`nav-item ${
                  activePage === item.name ? "active" : ""
                }`}
                onClick={() =>
                  setActivePage(item.name)
                }
              >
                <span className="nav-icon">
                  {item.icon}
                </span>

                <span>{item.name}</span>
              </button>
            ))}
          </nav>
        </div>

        <div className="sidebar-footer">
          <div className="system-status">
            <span className="status-dot" />

            <div>
              <strong>Data Available</strong>
              <small>Chicago crime snapshot</small>
            </div>
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div>
            <span className="breadcrumb">
              GARUDA / INTELLIGENCE
            </span>

            <h2>{activePage}</h2>
          </div>

          <div className="topbar-actions">
            <div className={`live-status live-status-${apiStatus}`} title={dataError || "API health is checked every 30 seconds"}>
              <span className="live-status-dot" />
              <span>{apiStatus === "checking" ? "CONNECTING" : apiStatus === "online" ? "LIVE API" : apiStatus === "degraded" ? "DEGRADED" : "API OFFLINE"}</span>
              {lastSync && <small>{lastSync.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</small>}
            </div>
            <button className="search-button">
              <span>⌕</span>
              <span>Search location</span>
              <kbd>⌘ K</kbd>
            </button>

            <button className="icon-button">
              ◇
            </button>

            <div className="profile">
              <div className="profile-avatar">
                G
              </div>

              <div>
                <strong>GARUDA</strong>
                <span>Analyst</span>
              </div>
            </div>
          </div>
        </header>

{activePage === "Overview" && (
          <>
        <section className="hero">
          <div className="hero-copy">
            <span className="eyebrow">
              PUBLIC SAFETY INTELLIGENCE
            </span>

            <h3>
              See the city.
              <br />
              <span>Follow the signal.</span>
            </h3>

            <p>
              Explore observed crime activity across
              Chicago and investigate locations through
              GARUDA&apos;s evidence-driven risk framework.
            </p>
          </div>

          <div className="hero-meta">
            <div className="hero-meta-item">
              <strong>50,000</strong>
              <span>OBSERVED RECORDS</span>
            </div>

            <div className="hero-divider" />

            <div className="hero-meta-item">
              <strong>274</strong>
              <span>BEATS</span>
            </div>

            <div className="hero-divider" />

            <div className="hero-meta-item">
              <strong>2025</strong>
              <span>DATA SNAPSHOT</span>
            </div>
          </div>
        </section>

          </>
        )}

{["Overview", "Risk Map", "Hotspots"].includes(activePage) && (
          <>
        <section className="map-stage">
          <div className="map-topbar">
            <div>
              <span className="map-kicker">
                GEOSPATIAL INTELLIGENCE
              </span>

              <h4>Chicago Crime Activity</h4>
            </div>

            <div className="map-controls">
              <button className="map-control active">
                HOTSPOTS
              </button>

              <button className="map-control">
                OBSERVED
              </button>

              <button className="map-control">
                RISK
              </button>
            </div>
          </div>

          <div className="map-wrapper">
            <MapContainer
              center={[41.84, -87.68]}
              zoom={10.8}
              zoomControl={false}
              scrollWheelZoom
              className="crime-map"
            >
              <TileLayer
                attribution="&copy; OpenStreetMap contributors"
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />

              {selectedPosition && (
                <>
                  <FlyToLocation
                    location={selectedPosition}
                  />

                  <InvestigationPosition
                    location={selectedPosition}
                    onPositionChange={
                      handlePositionChange
                    }
                  />
                </>
              )}

              {HOTSPOTS.map((spot) => {
                const isSelected =
                  selected?.beat === spot.beat;

                return (
                  <Fragment key={spot.beat}>
                    {isSelected && (
                      <CircleMarker
                        center={[
                          spot.lat,
                          spot.lon,
                        ]}
                        radius={24}
                        pathOptions={{
                          color: "#d9784f",
                          fillColor: "transparent",
                          fillOpacity: 0,
                          weight: 2,
                          opacity: 0.7,
                          className:
                            "selected-hotspot-ring",
                        }}
                      />
                    )}

                    <CircleMarker
                      center={[
                        spot.lat,
                        spot.lon,
                      ]}
                      radius={
                        isSelected
                          ? 13
                          : Math.max(
                              6,
                              spot.crimes / 65
                            )
                      }
                      pathOptions={{
                        color: isSelected
                          ? "#d9784f"
                          : "#a56a51",
                        fillColor: isSelected
                          ? "#b94e35"
                          : "#9a5d46",
                        fillOpacity: isSelected
                          ? 0.88
                          : 0.48,
                        weight: isSelected ? 3 : 1,
                        className: isSelected
                          ? "selected-hotspot-core"
                          : "",
                      }}
                      eventHandlers={{
                        click: () =>
                          handleSelect(spot),
                      }}
                    >
                      <Popup>
                        <strong>
                          Beat {spot.beat}
                        </strong>
                        <br />
                        {spot.crimes} observed
                        incidents
                      </Popup>
                    </CircleMarker>
                  </Fragment>
                );
              })}
            </MapContainer>

            <div className="map-overlay-grid" />

            <div className="map-location-count">
              <span className="location-pulse" />

              <div>
                <strong>
                  {HOTSPOTS.length}
                </strong>

                <span>
                  VISIBLE HOTSPOTS
                </span>
              </div>
            </div>

            <div className="map-legend">
              <span className="legend-title">
                ACTIVITY
              </span>

              <div>
                <i className="legend-low" />
                <span>LOW</span>
              </div>

              <div>
                <i className="legend-medium" />
                <span>MEDIUM</span>
              </div>

              <div>
                <i className="legend-high" />
                <span>HIGH</span>
              </div>
            </div>

            {selected &&
              investigationPosition && (
                <>
                  <div
                    className={`investigation-anchor ${
                      investigating ? "expanded" : ""
                    }`}
                    style={{
                      left:
                        investigationPosition.anchorLeft,
                      top:
                        investigationPosition.anchorTop,
                    }}
                  >
                    <div className="anchor-pulse" />
                    <div className="anchor-core" />
                  </div>

                  <div
                    className={`investigation-card ${
                      investigating ? "expanded" : ""
                    }`}
                    style={{
                      left:
                        investigationPosition.cardLeft,
                      top:
                        investigationPosition.cardTop,
                      right: "auto",
                    }}
                  >
                    <div className="investigation-line" />

                    <button
                      className="investigation-close"
                      onClick={handleClose}
                      aria-label="Close investigation"
                    >
                      ×
                    </button>

                    <span className="investigation-kicker">
                      LOCATION INVESTIGATION
                    </span>

                    <div className="investigation-title">
                      <div>
                        <span>BEAT</span>

                        <strong>
                          {selected.beat}
                        </strong>
                      </div>

                      <div className="risk-badge">
                        {prediction?.result?.consensus
                          ?.risk_level ||
                          "PENDING"}
                      </div>
                    </div>

                    <div className="investigation-count">
                      <strong>
                        {selected.crimes}
                      </strong>

                      <span>
                        OBSERVED INCIDENTS
                      </span>
                    </div>

                    <div className="investigation-divider" />

                    {!investigating ? (
                      <>
                        <p>
                          Explore the evidence signals
                          associated with this location.
                        </p>

                        <button
                          className="investigate-button"
                          onClick={
                            handleInvestigate
                          }
                          disabled={
                            predictionLoading
                          }
                        >
                          {predictionLoading
                            ? "ANALYZING..."
                            : "INVESTIGATE"}

                          <span>→</span>
                        </button>
                      </>
                    ) : (
                      <div className="evidence-stack">
                        {predictionLoading && (
                          <div className="prediction-status">
                            GARUDA IS CROSS-CHECKING
                            EVIDENCE...
                          </div>
                        )}

                        {predictionError && (
                          <div className="prediction-error">
                            {predictionError}
                          </div>
                        )}

                        {prediction && (
                          <>
                            <div className="evidence-item">
                              <div>
                                <span>
                                  ML PROBABILITY
                                </span>

                                <strong>
                                  {getSignalPercent(
                                    prediction.result
                                      .signals
                                      .ml_probability
                                  )}
                                  %
                                </strong>
                              </div>

                              <div className="evidence-track">
                                <span
                                  style={{
                                    width: `${getSignalPercent(
                                      prediction.result
                                        .signals
                                        .ml_probability
                                    )}%`,
                                  }}
                                />
                              </div>
                            </div>

                            <div className="evidence-item">
                              <div>
                                <span>
                                  TEMPORAL
                                </span>

                                <strong>
                                  {getSignalPercent(
                                    prediction.result
                                      .signals
                                      .temporal_consistency
                                  )}
                                  %
                                </strong>
                              </div>

                              <div className="evidence-track">
                                <span
                                  style={{
                                    width: `${getSignalPercent(
                                      prediction.result
                                        .signals
                                        .temporal_consistency
                                    )}%`,
                                  }}
                                />
                              </div>
                            </div>

                            <div className="evidence-item">
                              <div>
                                <span>
                                  ANOMALY
                                </span>

                                <strong>
                                  {getAnomalyPercent()}%
                                </strong>
                              </div>

                              <div className="evidence-track">
                                <span
                                  style={{
                                    width: `${getAnomalyPercent()}%`,
                                  }}
                                />
                              </div>
                            </div>

                            <div className="evidence-item">
                              <div>
                                <span>
                                  SPATIAL
                                </span>

                                <strong>
                                  {getSpatialPercent()}%
                                </strong>
                              </div>

                              <div className="evidence-track">
                                <span
                                  style={{
                                    width: `${getSpatialPercent()}%`,
                                  }}
                                />
                              </div>
                            </div>

                            <div className="consensus-result">
                              <span>
                                GARUDA CONSENSUS
                              </span>

                              <strong>
                                {getSignalPercent(
                                  prediction.result
                                    .consensus
                                    .consensus_score
                                )}
                                %
                              </strong>

                              <small>
                                {
                                  prediction.result
                                    .consensus
                                    .risk_level
                                }{" "}
                                RISK
                              </small>
                            </div>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                </>
              )}
          </div>

          <div className="map-footer">
            <span>
              Source: Chicago crime dataset · Observed
              records
            </span>

            <span>
              Click a hotspot to investigate
            </span>
          </div>
        </section>
          </>
        )}


        {activePage === "Crime Analysis" && (
          <section style={pagePanelStyle}>
            <PageHeading eyebrow="OBSERVED DATA" title="Crime Analysis"
              subtitle="Explore the available Chicago crime snapshot. These summaries describe observed records, not future crime." />
            <div style={metricGridStyle}>
              <Metric label="Observed records" value={datasetSummary ? Number(datasetSummary.rows).toLocaleString() : "Loading…"} />
              <Metric label="Police beats" value={datasetSummary ? Number(datasetSummary.unique_beats ?? 0).toLocaleString() : "Loading…"} />
              <Metric label="Date range" value={datasetSummary?.date_range ? `${String(datasetSummary.date_range.start).slice(0, 10)} – ${String(datasetSummary.date_range.end).slice(0, 10)}` : "Loading…"} />
              <Metric label="Model" value="XGBoost E4" />
            </div>
            <div className="analysis-grid">
              <section className="live-chart-card">
                <div className="live-chart-heading"><div><span className="eyebrow">CATEGORY DISTRIBUTION</span><h3>Most reported crime types</h3></div><span className="chart-live-tag"><i /> LIVE API</span></div>
                {crimeAnalysis?.crime_categories?.length ? <div className="trend-list">{crimeAnalysis.crime_categories.map((item, index) => { const max = Math.max(1, ...crimeAnalysis.crime_categories.map(x => Number(x.count || 0))); return <div className="trend-row" key={item.category}><span title={item.category}>{item.category}</span><div className="trend-track"><i style={{ width: `${Math.max(1, Number(item.count) / max * 100)}%`, animationDelay: `${index * 45}ms` }} /></div><strong>{Number(item.count).toLocaleString()}</strong></div>; })}</div> : <div className="chart-empty">{dataError || "Loading category totals from the backend…"}</div>}
                <p className="chart-footnote">Counts are grouped from the complete locally loaded dataset, not just the visible map hotspots.</p>
              </section>
              <section className="live-chart-card">
                <div className="live-chart-heading"><div><span className="eyebrow">BEAT COMPARISON</span><h3>Highest observed counts</h3></div><span className="chart-live-tag"><i /> LIVE API</span></div>
                {crimeAnalysis?.top_beats?.length ? <div className="trend-list">{crimeAnalysis.top_beats.map((item, index) => { const max = Math.max(1, ...crimeAnalysis.top_beats.map(x => Number(x.count || 0))); return <div className="trend-row" key={`${item.beat}-${index}`}><span>Beat {item.beat}</span><div className="trend-track"><i style={{ width: `${Math.max(1, Number(item.count) / max * 100)}%`, animationDelay: `${index * 45}ms` }} /></div><strong>{Number(item.count).toLocaleString()}</strong></div>; })}</div> : <div className="chart-empty">{dataError || "Loading beat totals from the backend…"}</div>}
                <p className="chart-footnote">Descriptive record counts only; this is not a validated future-risk ranking.</p>
              </section>
            </div>
            <div style={infoCardStyle}><h3>Data quality & coverage</h3><p>{crimeAnalysis ? `${Number(crimeAnalysis.valid_dated_records || 0).toLocaleString()} records have valid dates in the loaded snapshot.` : "Loading date quality…"}</p><p>Source: local Chicago crime dataset. The dashboard distinguishes observed records from model-generated experimental risk signals.</p></div>
          </section>
        )}

        {activePage === "Temporal Analysis" && (
          <section style={pagePanelStyle}>
            <PageHeading eyebrow="TIME-BASED EVIDENCE" title="Temporal Analysis"
              subtitle="Temporal consistency is one input to the consensus layer; it is not a standalone forecast." />
            <div style={metricGridStyle}>
              <Metric label="Signal source" value="7-day window" />
              <Metric label="Historical context" value="30 days" />
              <Metric label="Output range" value="0–100%" />
            </div>
            <div className="live-chart-card">
              <div className="live-chart-heading">
                <div><span className="eyebrow">OBSERVED INCIDENT TREND</span><h3>{temporalGranularity[0].toUpperCase() + temporalGranularity.slice(1)} activity</h3></div>
                <span className="chart-live-tag"><i /> {apiStatus === "online" ? "LIVE API" : apiStatus === "offline" ? "API OFFLINE" : "CONNECTING"}</span>
              </div>
              <div className="temporal-controls" role="group" aria-label="Temporal aggregation">
                {["daily", "weekly", "monthly"].map((option) => (
                  <button key={option} type="button" className={`temporal-control ${temporalGranularity === option ? "active" : ""}`} onClick={() => setTemporalGranularity(option)} aria-pressed={temporalGranularity === option}>
                    {option[0].toUpperCase() + option.slice(1)}
                  </button>
                ))}
              </div>
              {temporalData.length > 1 ? (() => {
                const points = temporalData.slice(temporalGranularity === "daily" ? -14 : -12);
                const values = points.map((point) => Number(point.crime_count ?? point.count ?? point.value ?? 0));
                const maxValue = Math.max(1, ...values);
                const total = values.reduce((sum, value) => sum + value, 0);
                const average = total / Math.max(1, values.length);
                const latest = values[values.length - 1] || 0;
                const previous = values.length > 1 ? values[values.length - 2] : null;
                const change = previous && previous !== 0 ? ((latest - previous) / previous) * 100 : null;
                const endDate = datasetSummary?.date_range?.end ? new Date(`${String(datasetSummary.date_range.end).slice(0, 10)}T00:00:00`) : null;
                const lastLabel = String(points[points.length - 1]?.period ?? "");
                let partialLastPeriod = false;
                if (endDate && temporalGranularity === "weekly" && /^\d{4}-\d{2}-\d{2}$/.test(lastLabel)) {
                  const start = new Date(`${lastLabel}T00:00:00`);
                  const end = new Date(start); end.setDate(end.getDate() + 6);
                  partialLastPeriod = endDate < end;
                } else if (endDate && temporalGranularity === "monthly" && /^\d{4}-\d{2}$/.test(lastLabel)) {
                  const [year, month] = lastLabel.split("-").map(Number);
                  const monthEnd = new Date(year, month, 0);
                  partialLastPeriod = endDate < monthEnd;
                }
                return <>
                  <div className="temporal-summary-grid">
                    <div><span>INCIDENTS IN VIEW</span><strong>{total.toLocaleString()}</strong></div>
                    <div><span>AVERAGE / PERIOD</span><strong>{Math.round(average).toLocaleString()}</strong></div>
                    <div><span>LATEST PERIOD CHANGE</span><strong className={change === null ? "" : change > 0 ? "change-up" : change < 0 ? "change-down" : ""}>{change === null ? "—" : `${change > 0 ? "+" : ""}${change.toFixed(1)}%`}</strong></div>
                  </div>
                  {partialLastPeriod && <div className="partial-period-note">Partial final {temporalGranularity.slice(0, -2)} period: the dataset ends before this period is complete. Compare with caution.</div>}
                  <div className="trend-list">
                    {points.map((point, index) => {
                      const label = point.period ?? point.week ?? point.date ?? `Period ${index + 1}`;
                      const value = Number(point.crime_count ?? point.count ?? point.value ?? 0);
                      return <div className="trend-row" key={`${label}-${index}`} title={`${label}: ${value.toLocaleString()} observed records`}>
                        <span>{String(label).slice(0, 10)}</span>
                        <div className="trend-track"><i style={{ width: `${Math.max(1, value / maxValue * 100)}%`, animationDelay: `${index * 35}ms` }} /></div>
                        <strong>{value.toLocaleString()}</strong>
                      </div>;
                    })}
                  </div>
                </>;
              })() : <div className="chart-empty">{apiStatus === "offline" ? "API unavailable — start the backend to load temporal data." : `Loading ${temporalGranularity} crime activity from the API…`}</div>}
              <p className="chart-footnote">Historical reported incidents only. Period changes describe the selected records and are not a forecast of future crime.</p>
            </div>
            <div style={infoCardStyle}>
              <h3>How to read the signal</h3>
              <p>The investigation panel shows the API's experimental temporal-consistency signal. Interpret it alongside ML, anomaly, and spatial signals.</p>
              <p><strong>Model limitation:</strong> the input-based API temporal calculation has not yet been confirmed equivalent to the history-based feature used during training.</p>
            </div>
          </section>
        )}

        {activePage === "Evidence" && (
          <section style={pagePanelStyle}>
            <PageHeading eyebrow="EXPLAINABILITY" title="Evidence"
              subtitle="Review the evidence components used by the current investigation workflow." />
            <div style={metricGridStyle}>
              <Metric label="Machine learning" value="XGBoost" />
              <Metric label="Anomaly detection" value="Isolation Forest" />
              <Metric label="Spatial analysis" value="DBSCAN" />
              <Metric label="Consensus layer" value="Weighted score" />
            </div>
            <div style={infoCardStyle}>
              <h3>Interpretation safeguards</h3>
              <p>Signals are experimental indicators derived from the supplied features. An anomaly score of 100% does not mean a 100% chance of crime.</p>
              <p>The consensus score combines the current signal outputs into a LOW, MEDIUM, or HIGH category. It should not be treated as proof about individuals or as a definitive prediction of future incidents.</p>
            </div>
          </section>
        )}

        {activePage === "Datasets" && (
          <section style={pagePanelStyle}>
            <PageHeading eyebrow="DATA CATALOG" title="Datasets"
              subtitle="Sources currently detected by the backend health endpoint." />
            <div style={metricGridStyle}>
              <Metric label="Chicago local dataset" value={healthData?.data_sources?.chicago_local_dataset ? "Available" : "Not detected"} />
              <Metric label="NCRB local dataset" value={healthData?.data_sources?.ncrb_local_dataset ? "Available" : "Not detected"} />
              <Metric label="UCI local dataset" value={healthData?.data_sources?.uci_local_dataset ? "Available" : "Not detected"} />
              <Metric label="UCI context artifact" value={healthData?.data_sources?.uci_context_artifact ? "Available" : "Not detected"} />
            </div>
            <div style={infoCardStyle}>
              <h3>Chicago crime snapshot</h3>
              <p>The dashboard's visible hotspot list is a selected set of 10 beats. The interface also reports a snapshot of 50,000 observed records across 274 beats for 2025.</p>
              <p>Dataset availability means files were detected by the API; it does not by itself certify completeness, freshness, or suitability for every analysis.</p>
            </div>
          </section>
        )}

        {activePage === "Research" && (
          <section style={pagePanelStyle}>
            <PageHeading eyebrow="METHODS & VALIDATION" title="Research"
              subtitle="GARUDA combines a classifier with temporal, anomaly, spatial, and consensus signals." />
            <div style={metricGridStyle}>
              <Metric label="Base classifier" value="XGBoost E4" />
              <Metric label="Anomaly model" value="Isolation Forest" />
              <Metric label="Spatial method" value="DBSCAN" />
              <Metric label="Output" value="Risk category" />
            </div>
            <div style={infoCardStyle}>
              <h3>Validation status</h3>
              <p>The API and dashboard are connected and returning responses. That verifies the software path, not the predictive quality of the model.</p>
              <p>Prior experiments showed modest discrimination, so claims about accuracy or operational effectiveness should be based on the evaluation metrics and limitations documented in the research artifacts.</p>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;