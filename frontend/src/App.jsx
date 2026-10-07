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

function App() {
  const [activePage, setActivePage] = useState("Overview");
  const [selected, setSelected] = useState(null);
  const [investigating, setInvestigating] = useState(false);
  const [investigationPosition, setInvestigationPosition] =
    useState(null);

  const [prediction, setPrediction] = useState(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState("");

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
      </main>
    </div>
  );
}

export default App;