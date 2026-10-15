import {
  startTransition,
  useDeferredValue,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  Activity,
  ArrowRight,
  BarChart3,
  ChevronDown,
  Database,
  History,
  Menu,
  Play,
  Search,
  ShieldCheck,
  Target,
  TrendingUp,
  Trophy,
  Users,
  X,
} from "lucide-react";
import "./App.css";
import {
  api,
  STATIC_DATA,
  type ApiHealth,
  type InjuryTag,
  type Matchup,
  type ModelMetrics,
  type Player,
  type Prediction,
  type PredictionHistory,
  type Position,
  type PositionPrediction,
  type ROSRanking,
  type WeeklyRanking,
} from "./api";

type Page = "prediction" | "players" | "ros" | "history" | "models";
const NAV_ITEMS: Array<{ id: Page; label: string; icon: typeof Target }> = [
  { id: "prediction", label: "Prediction", icon: Target },
  { id: "players", label: "Players", icon: Users },
  { id: "ros", label: "Rankings", icon: Trophy },
  { id: "history", label: "History", icon: History },
  { id: "models", label: "Models", icon: BarChart3 },
];
const POSITIONS: Position[] = ["WR", "QB", "RB", "TE"];
const DEFAULT_PLAYER: Record<Position, string> = {
  WR: "DeVonta Smith",
  QB: "Josh Allen",
  RB: "Bijan Robinson",
  TE: "Trey McBride",
};
const teamColor: Record<string, string> = {
  ARI: "#97233f",
  ATL: "#a71930",
  BAL: "#241773",
  CAR: "#0085ca",
  CIN: "#fb4f14",
  CLE: "#311d00",
  DAL: "#003594",
  DEN: "#fb4f14",
  DET: "#0076b6",
  HOU: "#03202f",
  IND: "#002c5f",
  JAX: "#006778",
  KC: "#e31837",
  LA: "#003594",
  LAC: "#0080c6",
  LAR: "#003594",
  LV: "#000000",
  MIA: "#008e97",
  MIN: "#4f2683",
  NE: "#002244",
  NO: "#d3bc8d",
  NYG: "#0b2265",
  NYJ: "#125740",
  PHI: "#004c54",
  PIT: "#ffb612",
  SEA: "#002244",
  SF: "#aa0000",
  TB: "#d50a0a",
  TEN: "#4b92db",
  WAS: "#5a1414",
  WSH: "#5a1414",
  CHI: "#0b162a",
  BUF: "#00338d",
  GB: "#203731",
};

function TeamMark({
  team,
  size = "md",
}: {
  team: string;
  size?: "sm" | "md" | "lg";
}) {
  return (
    <span
      className={`team-mark team-mark-${size}`}
      style={{ backgroundColor: teamColor[team] ?? "#354457" }}
      aria-label={`${team} team mark`}
    >
      {team}
    </span>
  );
}
function CompactSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: number;
  options: number[];
  onChange: (value: number) => void;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const closeOnOutsideClick = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    window.addEventListener("pointerdown", closeOnOutsideClick);
    return () => window.removeEventListener("pointerdown", closeOnOutsideClick);
  }, [open]);

  return (
    <div
      className={`select-wrap ${open ? "open" : ""}`}
      ref={rootRef}
      onKeyDown={(event) => {
        if (event.key === "Escape") setOpen(false);
      }}
    >
      <button
        type="button"
        className="select-trigger"
        aria-label={label}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
      >
        <span>{value}</span>
        <ChevronDown size={15} />
      </button>
      {open && (
        <div className="select-menu" role="listbox" aria-label={label}>
          {options.map((option) => (
            <button
              type="button"
              role="option"
              aria-selected={option === value}
              className={option === value ? "selected" : ""}
              key={option}
              onClick={() => {
                onChange(option);
                setOpen(false);
              }}
            >
              {option}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
function PlayerPortrait({
  player,
  large = false,
}: {
  player: Player;
  large?: boolean;
}) {
  if (player.headshot_url)
    return (
      <img
        className={`portrait ${large ? "portrait-large" : ""}`}
        src={player.headshot_url}
        alt=""
      />
    );
  const initials = player.name
    .split(" ")
    .map((part) => part[0])
    .slice(0, 2)
    .join("");
  return (
    <span
      className={`portrait portrait-fallback ${large ? "portrait-large" : ""}`}
    >
      {initials}
    </span>
  );
}
function PlayerSearch({
  season,
  position,
  onSelect,
}: {
  season: number;
  position: Position;
  onSelect: (player: Player) => void;
}) {
  const [query, setQuery] = useState("");
  const [players, setPlayers] = useState<Player[]>([]);
  const [open, setOpen] = useState(false);
  const deferredQuery = useDeferredValue(query);
  useEffect(() => {
    let active = true;
    api
      .players(deferredQuery, season, 12, position)
      .then((data) => active && setPlayers(data))
      .catch(() => active && setPlayers([]));
    return () => {
      active = false;
    };
  }, [deferredQuery, position, season]);
  return (
    <div className="player-search">
      <Search size={18} />
      <input
        value={query}
        onChange={(event) => {
          setQuery(event.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        placeholder="Search players"
        aria-label="Search players"
      />
      {query && (
        <button
          className="icon-button"
          onClick={() => setQuery("")}
          aria-label="Clear search"
        >
          <X size={16} />
        </button>
      )}
      {open && players.length > 0 && (
        <div className="search-results">
          {players.map((player) => (
            <button
              key={player.player_id}
              className="search-result"
              onClick={() => {
                onSelect(player);
                setQuery(player.name);
                setOpen(false);
              }}
            >
              <PlayerPortrait player={player} />
              <span>
                <strong>{player.name}</strong>
                <small>{player.team} · {seasonSummary(player, position)}</small>
              </span>
              <ArrowRight size={15} />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
function PositionTabs({ value, onChange }: { value: Position; onChange: (position: Position) => void }) {
  return (
    <div className="position-tabs" aria-label="Position">
      {POSITIONS.map((position) => (
        <button key={position} className={value === position ? "active" : ""} onClick={() => onChange(position)}>
          {position}
        </button>
      ))}
    </div>
  );
}
function MatchupCard({
  player,
  matchup,
}: {
  player: Player | null;
  matchup: Matchup | null;
}) {
  if (!player || !matchup)
    return <div className="matchup-card skeleton-block" />;
  const homeTeam = matchup.home_game ? player.team : matchup.opponent;
  const awayTeam = matchup.home_game ? matchup.opponent : player.team;
  return (
    <div className="matchup-card">
      <div className="matchup-team">
        <TeamMark team={awayTeam} size="lg" />
        <strong>{awayTeam}</strong>
        <span>Away</span>
      </div>
      <div className="matchup-center">
        <small>WEEK {matchup.week}</small>
        <b>AT</b>
        <span>
          {formatDate(matchup.game_date)} · {matchup.game_time ?? "TBD"}
        </span>
      </div>
      <div className="matchup-team">
        <TeamMark team={homeTeam} size="lg" />
        <strong>{homeTeam}</strong>
        <span>Home</span>
      </div>
    </div>
  );
}
function TrendBars({
  label,
  values,
  projected,
  decimals = 0,
}: {
  label: string;
  values: Array<{ week: number; game_id: string; value: number }>;
  projected?: number;
  decimals?: number;
}) {
  const max = Math.max(...values.map((game) => game.value), projected ?? 0, 1);
  return (
    <div className="trend-metric-block">
      <span className="trend-metric-label">{label}</span>
      <div className="trend-chart" aria-label={`Recent ${label.toLowerCase()} trend`}>
      {values.map((game) => (
        <div className="trend-column" key={game.game_id}>
          <span className="trend-value">{game.value.toFixed(decimals)}</span>
          <div className="trend-bar-wrap">
            <span
              className="trend-bar"
              style={{ height: `${Math.max(10, (game.value / max) * 100)}%` }}
            />
          </div>
          <small>W{game.week}</small>
        </div>
      ))}
      {projected != null && <div className="trend-column projected">
        <span className="trend-value">
          {projected.toFixed(decimals)}
        </span>
        <div className="trend-bar-wrap">
          <span
            className="trend-bar"
            style={{ height: `${(projected / max) * 100}%` }}
          />
        </div>
        <small>PROJ</small>
      </div>}
    </div>
    </div>
  );
}
function TrendChart({ prediction }: { prediction: Prediction }) {
  const games = prediction.recent_games;
  if (!games.length) return <div className="empty-chart">No completed games this season.</div>;
  return (
    <div className="trend-metrics">
      <TrendBars
        label="Receiving yards"
        values={games.map((game) => ({ week: game.week, game_id: game.game_id, value: game.yards }))}
        projected={prediction.predicted_yards}
        decimals={0}
      />
      <TrendBars
        label="Receptions"
        values={games.map((game) => ({ week: game.week, game_id: game.game_id, value: game.receptions }))}
        projected={prediction.expected_receptions}
        decimals={1}
      />
    </div>
  );
}
function ModelGauge({
  label,
  value,
  active,
  unit = "YDS",
}: {
  label: string;
  value: number;
  active?: boolean;
  unit?: string;
}) {
  return (
    <div className={`model-gauge ${active ? "active" : ""}`}>
      <div className="model-gauge-head">
        <span>{label}</span>
        {active && <span className="primary-tag">Primary</span>}
      </div>
      <strong>
        {value.toFixed(1)} <small>{unit}</small>
      </strong>
      <div className="gauge-track">
        <span style={{ width: `${Math.min(100, (value / 130) * 100)}%` }} />
      </div>
    </div>
  );
}
function PredictionOutput({ prediction }: { prediction: Prediction }) {
  const features = prediction.features;
  const projectedPpr = prediction.expected_receptions + prediction.predicted_yards / 10 + prediction.expected_receiving_touchdowns * 6;
  return (
    <section className="output-section reveal">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Model output</span>
          <h2>Receiving yards forecast</h2>
        </div>
        <span className="generated-at">
          Updated{" "}
          {new Date(prediction.generated_at).toLocaleTimeString([], {
            hour: "numeric",
            minute: "2-digit",
          })}
        </span>
      </div>
      <div className="output-grid">
        <article className="primary-output">
          <span>Primary projection</span>
          <div className="primary-metric-row">
            <div>
              <div className="hero-number">
                {prediction.predicted_yards.toFixed(1)}
              </div>
              <b>receiving yards</b>
            </div>
            <div>
              <div className="hero-number">
                {projectedPpr.toFixed(1)}
              </div>
              <b>Projected PPR</b>
            </div>
          </div>
          <small>Historical MAE ≈ 20.1 yards</small>
          <div className="volume-projections">
            <div>
              <span>Expected targets</span>
              <strong>{formatNumber(prediction.expected_targets)}</strong>
              <small>MAE 1.86</small>
            </div>
            <div>
              <span>Expected receptions</span>
              <strong>{formatNumber(prediction.expected_receptions)}</strong>
              <small>MAE 1.38</small>
            </div>
            <div>
              <span>Expected receiving TDs</span>
              <strong>
                {formatNumber(prediction.expected_receiving_touchdowns)}
              </strong>
              <small>
                {formatPercent(prediction.anytime_touchdown_probability)} anytime
              </small>
            </div>
          </div>
        </article>
        <article className="range-output">
          <span>80% forecast interval</span>
          <strong>
            {prediction.model_range.low.toFixed(1)} –{" "}
            {prediction.model_range.high.toFixed(1)}
          </strong>
          <small>10th to 90th percentile outcome range</small>
          <div className="range-line">
            <i
              style={{
                left: `${prediction.model_range.low / 1.3}%`,
                width: `${(prediction.model_range.high - prediction.model_range.low) / 1.3}%`,
              }}
            />
          </div>
        </article>
        <div className="model-stack">
          <ModelGauge
            label="Weighted blend"
            value={prediction.ensemble_prediction}
            active
          />
          <ModelGauge label="Ridge" value={prediction.predictions.ridge} />
          <ModelGauge label="XGBoost" value={prediction.predictions.xgboost} />
          <ModelGauge
            label="Neural MLP"
            value={prediction.predictions.numerical_mlp}
          />
        </div>
      </div>
      <div className="insight-grid">
        <article className="data-panel">
          <div className="panel-title">
            <TrendingUp size={17} />
            <h3>Recent trend</h3>
          </div>
          <TrendChart prediction={prediction} />
        </article>
        <article className="data-panel">
          <div className="panel-title">
            <ShieldCheck size={17} />
            <h3>Role &amp; matchup context</h3>
          </div>
          <div className="stat-list">
            <div>
              <span>Current depth role</span>
              <strong>{formatDepthRole(features.depth_chart_rank)}</strong>
            </div>
            <div>
              <span>Healthy role baseline</span>
              <strong>{formatDepthRole(features.depth_chart_rank_healthy_baseline)}</strong>
            </div>
            <div>
              <span>Target share · last 5</span>
              <strong>{formatPercent(features.target_share_avg_last_5)}</strong>
            </div>
            <div>
              <span>Target share trend</span>
              <strong>{formatPercent(features.target_share_delta_last_3)}</strong>
            </div>
            <div>
              <span>Offseason vacated target share</span>
              <strong>{formatPercent(features.vacated_team_target_share)}</strong>
            </div>
            <div>
              <span>Neutral pass rate · last 5</span>
              <strong>{formatPercent(features.neutral_pass_rate_avg_last_5)}</strong>
            </div>
            <div>
              <span>Team WR target share · last 5</span>
              <strong>
                {formatPercent(features.team_wr_target_share_avg_last_5)}
              </strong>
            </div>
            <div>
              <span>Questionable teammate target share</span>
              <strong>
                {formatPercent(features.teammate_questionable_target_share)}
              </strong>
            </div>
            <div>
              <span>Implied team points</span>
              <strong>{formatNumber(features.team_implied_points)}</strong>
            </div>
            <div>
              <span>Average depth of target · last 5</span>
              <strong>{formatNumber(features.adot_avg_last_5)}</strong>
            </div>
            <div>
              <span>Games with team</span>
              <strong>{featureValue(features, "games_with_current_team")}</strong>
            </div>
            <div>
              <span>Opponent WR yards allowed · last 5</span>
              <strong>
                {formatNumber(prediction.opponent_context.yards_allowed_avg_last_5)}
              </strong>
            </div>
          </div>
        </article>
      </div>
    </section>
  );
}
const statLabels: Record<string, string> = {
  attempts: "Pass attempts",
  passing_yards: "Passing yards",
  passing_tds: "Passing TDs",
  passing_interceptions: "Interceptions",
  carries: "Carries",
  rushing_yards: "Rushing yards",
  rushing_tds: "Rushing TDs",
  targets: "Targets",
  receptions: "Receptions",
  receiving_yards: "Receiving yards",
  receiving_tds: "Receiving TDs",
};
function seasonSummary(player: Player, position: Position) {
  if (position === "QB") {
    return `${player.season_passing_yards ?? 0} pass YDS this season`;
  }
  if (position === "RB") {
    return `${player.season_rushing_yards ?? 0} rush YDS this season`;
  }
  return `${player.season_receiving_yards} rec YDS this season`;
}
function primaryPositionValue(prediction: PositionPrediction) {
  if (prediction.position === "QB") return prediction.predictions.passing_yards ?? 0;
  if (prediction.position === "RB") {
    return (prediction.predictions.rushing_yards ?? 0) + (prediction.predictions.receiving_yards ?? 0);
  }
  return prediction.predictions.receiving_yards ?? 0;
}
function primaryPositionLabel(position: Position) {
  if (position === "QB") return "Passing yards";
  if (position === "RB") return "Total yards";
  return "Receiving yards";
}
function primaryPositionUnit(position: Position) {
  if (position === "QB") return "PASS YARDS";
  if (position === "RB") return "TOTAL YARDS";
  return "RECEIVING YARDS";
}
function PositionTrendChart({ prediction }: { prediction: PositionPrediction }) {
  const projected = primaryPositionValue(prediction);
  if (!prediction.recent_games.length) {
    return <div className="empty-chart">No completed games this season.</div>;
  }
  return (
    <div className="trend-metrics">
      <TrendBars
        label={primaryPositionLabel(prediction.position)}
        values={prediction.recent_games.map((game) => ({ ...game, value: game.value }))}
        projected={projected}
        decimals={1}
      />
      {prediction.secondary_recent_label && (
        <TrendBars
          label={prediction.secondary_recent_label}
          values={prediction.recent_games.map((game) => ({
            ...game,
            value: game.secondary_value ?? 0,
          }))}
          projected={prediction.position === "QB" ? prediction.predictions.attempts : prediction.predictions.carries ?? 0}
          decimals={1}
        />
      )}
    </div>
  );
}
function PositionPredictionOutput({ prediction }: { prediction: PositionPrediction }) {
  if (prediction.position === "TE") {
    return <TEPredictionOutput prediction={prediction} />;
  }
  const entries = Object.entries(prediction.predictions);
  const primary = primaryPositionValue(prediction);
  return (
    <section className="output-section reveal">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Model output</span>
          <h2>{prediction.position} forecast</h2>
        </div>
      </div>
      <div className="te-output-grid">
        <article className="primary-output">
          <span>Primary projection</span>
          <div className="primary-metric-row">
            <div>
              <div className="hero-number">{primary.toFixed(1)}</div>
              <b>{primaryPositionUnit(prediction.position)}</b>
            </div>
            <div>
              <div className="hero-number">{prediction.projected_ppr_points.toFixed(1)}</div>
              <b>Projected PPR</b>
            </div>
          </div>
          <small>{modelLabel(prediction.selected_models[prediction.primary_target] ?? "model")}</small>
          <div className="volume-projections">
            {prediction.position === "QB" ? (
              <>
                <div><span>Pass attempts</span><strong>{formatNumber(prediction.predictions.attempts)}</strong><small>{modelLabel(prediction.selected_models.attempts ?? "model")}</small></div>
                <div><span>Passing TDs</span><strong>{formatNumber(prediction.predictions.passing_tds)}</strong><small>{modelLabel(prediction.selected_models.passing_tds ?? "model")}</small></div>
                <div><span>Rushing yards</span><strong>{formatNumber(prediction.predictions.rushing_yards)}</strong><small>{modelLabel(prediction.selected_models.rushing_yards ?? "model")}</small></div>
              </>
            ) : (
              <>
                <div><span>Carries</span><strong>{formatNumber(prediction.predictions.carries)}</strong><small>{modelLabel(prediction.selected_models.carries ?? "model")}</small></div>
                <div><span>Receptions</span><strong>{formatNumber(prediction.predictions.receptions)}</strong><small>{modelLabel(prediction.selected_models.receptions ?? "model")}</small></div>
              </>
            )}
          </div>
        </article>
        <article className="range-output">
          <span>80% forecast interval</span>
          <strong>{prediction.primary_quantiles.q10.toFixed(1)} – {prediction.primary_quantiles.q90.toFixed(1)}</strong>
          <small>10th to 90th percentile {primaryPositionLabel(prediction.position).toLowerCase()} range</small>
          <div className="range-line">
            <i
              style={{
                left: `${(prediction.primary_quantiles.q10 / Math.max(1, prediction.primary_quantiles.q90 * 1.25)) * 100}%`,
                width: `${((prediction.primary_quantiles.q90 - prediction.primary_quantiles.q10) / Math.max(1, prediction.primary_quantiles.q90 * 1.25)) * 100}%`,
              }}
            />
          </div>
        </article>
        <div className="model-stack te-model-stack">
          {(prediction.position === "QB" || prediction.position === "RB") && prediction.model_predictions
            ? ["weighted_blend", "ridge", "xgboost", "neural_mlp"].map((key, index) => (
                <ModelGauge key={key} label={key === "weighted_blend" ? "Weighted blend" : modelLabel(key)} value={prediction.model_predictions?.[key] ?? 0} active={index === 0} unit="YDS" />
              ))
            : entries.map(([key, value], index) => (
                <ModelGauge key={key} label={statLabels[key] ?? modelLabel(key)} value={value} active={index === 0} unit="" />
              ))}
        </div>
      </div>
      <PositionContextPanels prediction={prediction} />
    </section>
  );
}
function featureValue(features: Record<string, number | null>, key: string) {
  return formatNumber(features[key]);
}
function PositionContextPanels({ prediction }: { prediction: PositionPrediction }) {
  const { features, position } = prediction;
  const workloadRows = position === "QB"
    ? [
        ["Pass attempts · last 5", featureValue(features, "attempts_avg_last_5")],
        ["Pass yards · last 5", featureValue(features, "passing_yards_avg_last_5")],
        ["Carries · last 5", featureValue(features, "carries_avg_last_5")],
        ["Rush yards · last 5", featureValue(features, "rushing_yards_avg_last_5")],
      ]
    : position === "RB"
    ? [
        ["Depth chart role", formatDepthRole(features.depth_chart_rank)],
        ["Healthy role baseline", formatDepthRole(features.depth_chart_rank_healthy_baseline)],
        ["Snap share · last game", formatPercent(features.snap_share_last_game)],
        ["Carry share · last 5", formatPercent(features.rb_carry_share_avg_last_5)],
        ["Carry share trend", formatPercent(features.carry_share_delta_last_3)],
        ["Carries · last 5", featureValue(features, "carries_avg_last_5")],
      ]
    : [
        ["Depth chart role", formatDepthRole(features.depth_chart_rank)],
        ["Snap share · last game", formatPercent(features.snap_share_last_game)],
        ["Snap share · last 3", formatPercent(features.snap_share_avg_last_3)],
        ["Targets · last 5", featureValue(features, "targets_avg_last_5")],
      ];
  const contextRows = position === "QB"
    ? [
        ["Implied team points", featureValue(features, "team_implied_points")],
        ["Expected scoring margin", featureValue(features, "team_implied_margin")],
        ["Neutral pass rate · last 5", formatPercent(features.playcall_neutral_pass_rate_avg_last_5)],
        ["Trailing pass rate · last 5", formatPercent(features.playcall_trailing_pass_rate_avg_last_5)],
        ["Leading pass rate · last 5", formatPercent(features.playcall_leading_pass_rate_avg_last_5)],
        ["Late trailing pass rate · last 5", formatPercent(features.playcall_late_trailing_pass_rate_avg_last_5)],
        ["Head-coach neutral pass rate", formatPercent(features.coach_neutral_pass_rate_avg_last_16)],
        ["Offensive plays · last 5", featureValue(features, "playcall_plays_avg_last_5")],
        ["Team pass rate · last 5", formatPercent(features.team_pass_rate_avg_last_5)],
        ["Red-zone pass rate · last 5", formatPercent(features.team_red_zone_pass_rate_avg_last_5)],
        ["Opp pass yards allowed · last 5", featureValue(features, "opponent_passing_yards_allowed_avg_last_5")],
        ["Opp rush yards allowed · last 5", featureValue(features, "opponent_rushing_yards_allowed_avg_last_5")],
      ]
    : position === "RB"
    ? [
        ["Implied team points", featureValue(features, "team_implied_points")],
        ["Team rush rate · last 5", formatPercent(features.team_rush_rate_avg_last_5)],
        ["Red-zone rush rate · last 5", formatPercent(features.team_red_zone_rush_rate_avg_last_5)],
        ["Opp rush yards allowed · last 5", featureValue(features, "opponent_rushing_yards_allowed_avg_last_5")],
        ["Opp RB rec yards allowed · last 5", featureValue(features, "opponent_receiving_yards_allowed_avg_last_5")],
        ["Games with team", featureValue(features, "games_with_current_team")],
      ]
    : [
        ["Implied team points", featureValue(features, "team_implied_points")],
        ["Team pass rate · last 5", formatPercent(features.team_pass_rate_avg_last_5)],
        ["Red-zone pass rate · last 5", formatPercent(features.team_red_zone_pass_rate_avg_last_5)],
        ["Opp TE yards allowed · last 5", featureValue(features, "opponent_receiving_yards_allowed_avg_last_5")],
        ["Rec TDs · last 5", featureValue(features, "receiving_tds_avg_last_5")],
      ];
  return (
    <div className="insight-grid">
      <article className="data-panel">
        <div className="panel-title"><TrendingUp size={17} /><h3>Recent trend</h3></div>
        <PositionTrendChart prediction={prediction} />
      </article>
      <article className="data-panel">
        <div className="panel-title"><ShieldCheck size={17} /><h3>{position} context</h3></div>
        <div className="stat-list">
          {workloadRows.map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}
          {contextRows.map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}
        </div>
      </article>
    </div>
  );
}
function TEPredictionOutput({ prediction }: { prediction: PositionPrediction }) {
  const yards = prediction.predictions.receiving_yards ?? 0;
  const targets = prediction.predictions.targets ?? 0;
  const receptions = prediction.predictions.receptions ?? 0;
  const touchdowns = prediction.predictions.receiving_tds ?? 0;
  const ppr = receptions + yards / 10 + touchdowns * 6;
  return (
    <section className="output-section reveal">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Model output</span>
          <h2>Receiving yards forecast</h2>
        </div>
      </div>
      <div className="te-output-grid">
        <article className="primary-output">
          <span>Primary projection</span>
          <div className="primary-metric-row">
            <div>
              <div className="hero-number">{yards.toFixed(1)}</div>
              <b>receiving yards</b>
            </div>
            <div>
              <div className="hero-number">{ppr.toFixed(1)}</div>
              <b>Projected PPR</b>
            </div>
          </div>
          <small>{modelLabel(prediction.selected_models.receiving_yards ?? "model")}</small>
          <div className="volume-projections">
            <div>
              <span>Expected targets</span>
              <strong>{targets.toFixed(1)}</strong>
              <small>{modelLabel(prediction.selected_models.targets ?? "model")}</small>
            </div>
            <div>
              <span>Expected receptions</span>
              <strong>{receptions.toFixed(1)}</strong>
              <small>{modelLabel(prediction.selected_models.receptions ?? "model")}</small>
            </div>
            <div>
              <span>Expected receiving TDs</span>
              <strong>{touchdowns.toFixed(1)}</strong>
              <small>{modelLabel(prediction.selected_models.receiving_tds ?? "model")}</small>
            </div>
          </div>
        </article>
        <article className="range-output">
          <span>80% forecast interval</span>
          <strong>{prediction.primary_quantiles.q10.toFixed(1)} – {prediction.primary_quantiles.q90.toFixed(1)}</strong>
          <small>10th to 90th percentile receiving-yards range</small>
          <div className="range-line">
            <i
              style={{
                left: `${(prediction.primary_quantiles.q10 / Math.max(1, prediction.primary_quantiles.q90 * 1.25)) * 100}%`,
                width: `${((prediction.primary_quantiles.q90 - prediction.primary_quantiles.q10) / Math.max(1, prediction.primary_quantiles.q90 * 1.25)) * 100}%`,
              }}
            />
          </div>
        </article>
        <div className="model-stack te-model-stack">
          <ModelGauge label="Receiving yards" value={yards} active unit="YDS" />
          <ModelGauge label="Targets" value={targets} unit="TGT" />
          <ModelGauge label="Receptions" value={receptions} unit="REC" />
          <ModelGauge label="Receiving TDs" value={touchdowns} unit="TD" />
        </div>
      </div>
      <div className="insight-grid">
        <article className="data-panel">
          <div className="panel-title">
            <TrendingUp size={17} />
            <h3>Recent trend</h3>
          </div>
          <PositionTrendChart prediction={prediction} />
        </article>
        <article className="data-panel">
          <div className="panel-title">
            <ShieldCheck size={17} />
            <h3>Game context</h3>
          </div>
          <div className="stat-list">
            <div><span>Depth chart role</span><strong>{formatDepthRole(prediction.features.depth_chart_rank)}</strong></div>
            <div><span>Targets · last 5</span><strong>{featureValue(prediction.features, "targets_avg_last_5")}</strong></div>
            <div><span>Rec yards · last 5</span><strong>{featureValue(prediction.features, "receiving_yards_avg_last_5")}</strong></div>
            <div><span>Implied team points</span><strong>{featureValue(prediction.features, "team_implied_points")}</strong></div>
            <div><span>Team pass rate · last 5</span><strong>{formatPercent(prediction.features.team_pass_rate_avg_last_5)}</strong></div>
            <div><span>Red-zone pass rate · last 5</span><strong>{formatPercent(prediction.features.team_red_zone_pass_rate_avg_last_5)}</strong></div>
            <div><span>Opp TE yards allowed · last 5</span><strong>{featureValue(prediction.features, "opponent_receiving_yards_allowed_avg_last_5")}</strong></div>
            <div><span>Snap share · last game</span><strong>{formatPercent(prediction.features.snap_share_last_game)}</strong></div>
          </div>
        </article>
      </div>
    </section>
  );
}
type ForecastSelection = { playerId: string; position: Position; week: number };
type OpenForecast = (playerId: string, position: Position, week?: number) => void;
function PredictionPage({ onSaved, currentWeek, health, selection }: { onSaved: () => void; currentWeek: number; health: ApiHealth | null; selection: ForecastSelection | null }) {
  const [position, setPosition] = useState<Position>(selection?.position ?? "WR");
  const [season, setSeason] = useState(2026),
    [week, setWeek] = useState(selection?.week ?? currentWeek);
  const [player, setPlayer] = useState<Player | null>(null),
    [matchup, setMatchup] = useState<Matchup | null>(null),
    [prediction, setPrediction] = useState<Prediction | null>(null),
    [positionPrediction, setPositionPrediction] = useState<PositionPrediction | null>(null);
  const [loading, setLoading] = useState(false),
    [error, setError] = useState("");
  useEffect(() => setWeek(selection?.week ?? currentWeek), [currentWeek, selection]);
  useEffect(() => {
    let active = true;
    if (selection && position === selection.position) {
      api.player(selection.playerId, season)
        .then(next => { if (active) setPlayer({ ...next, position }); })
        .catch(cause => { if (active) setError(cause instanceof Error ? cause.message : "Player unavailable"); });
      return () => { active = false; };
    }
    api
      .players(DEFAULT_PLAYER[position], season, 12, position)
      .then(players => STATIC_DATA && !players.length ? api.players("", season, 1, position) : players)
      .then((players) => active && players[0] && setPlayer(players[0]))
      .catch(() => undefined);
    return () => { active = false; };
  }, [position, season, selection]);
  useEffect(() => {
    if (!player) return;
    api
      .matchup(player.player_id, season, week)
      .then(setMatchup)
      .catch(() => setMatchup(null));
  }, [player, season, week]);
  useEffect(() => {
    if ((!STATIC_DATA && !selection) || !player || player.position !== position) return;
    let active = true;
    setLoading(true);
    setError("");
    setPrediction(null);
    setPositionPrediction(null);
    const body = { player_name: player.name, team: player.team, season, week };
    const pending = position === "WR"
      ? api.predict({ ...body, exclude_current_season_data: false })
      : api.predictPosition({ ...body, position });
    pending.then(result => {
      if (!active) return;
      if (position === "WR") { setPrediction(result as Prediction); onSaved(); }
      else setPositionPrediction(result as PositionPrediction);
    }).catch(cause => { if (active) setError(cause instanceof Error ? cause.message : "Forecast unavailable"); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [player, position, season, week, health?.generated_at]);
  const runPrediction = async () => {
    if (!player) return;
    setLoading(true);
    setError("");
    setPrediction(null);
    setPositionPrediction(null);
    try {
      if (position === "WR") {
        const result = await api.predict({
          player_name: player.name,
          team: player.team,
          season,
          week,
          exclude_current_season_data: false,
        });
        startTransition(() => setPrediction(result));
        onSaved();
      } else {
        const result = await api.predictPosition({
          player_name: player.name,
          position,
          team: player.team,
          season,
          week,
        });
        startTransition(() => setPositionPrediction(result));
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  };
  return (
    <main className="page-content">
      <section className="page-intro">
        <div>
          <h1>Check the game script</h1>
          <p>
            {STATIC_DATA ? "Select a player to explore a published forecast. Updated daily at 3 AM Eastern—not live." : "Select a player and upcoming matchup. The Script uses only information available before kickoff."}
          </p>
          {STATIC_DATA && health?.generated_at && <p>Forecast updated {new Date(health.generated_at).toLocaleString()} · Data through week {health.latest_completed_week ?? "—"}{health.source_stale ? " · Saved data (source unavailable)" : ""}{health.preview_only ? " · Preview release" : ""}</p>}
        </div>
      </section>
      <section className="prediction-workbench">
        <div className={`control-strip ${STATIC_DATA ? "static-controls" : ""}`}>
          <div className="control-group position-control">
            <label>Position</label>
            <PositionTabs
              value={position}
              onChange={(next) => {
                setPosition(next);
                setPlayer(null);
                setPrediction(null);
                setPositionPrediction(null);
              }}
            />
          </div>
          <div className="control-group search-control">
            <label>Player</label>
            <PlayerSearch
              key={position}
              season={season}
              position={position}
              onSelect={(next) => {
                setPlayer(next);
                setPrediction(null);
                setPositionPrediction(null);
              }}
            />
          </div>
          <div className="control-group">
            <label>Season</label>
            <CompactSelect
              label="Season"
              value={season}
              options={STATIC_DATA ? [health?.season ?? 2026] : [2026, 2025]}
              onChange={setSeason}
            />
          </div>
          <div className="control-group">
            <label>Week</label>
            <CompactSelect
              label="Week"
              value={week}
              options={STATIC_DATA ? health?.available_weeks ?? [currentWeek] : Array.from(
                { length: Math.max(0, 19 - currentWeek) },
                (_, index) => index + currentWeek,
              )}
              onChange={(nextWeek) => {
                setWeek(nextWeek);
                setPrediction(null);
              }}
            />
          </div>
          {!STATIC_DATA && <button
            className="primary-button"
            disabled={!player || !matchup || loading}
            onClick={runPrediction}
          >
            {loading ? (
              <Activity className="spin" size={17} />
            ) : (
              <Play size={17} fill="currentColor" />
            )}
            {loading ? "Running models" : "Run prediction"}
          </button>}
        </div>
        <div className="selection-grid">
          <article className="player-summary">
            {player ? (
              <>
                <PlayerPortrait player={player} large />
                <div>
                  <span className="position-tag">{position}</span>
                  <h2>{player.name}</h2>
                  <p>
                    <TeamMark team={player.team} size="sm" /> {player.team} ·{" "}
                    {seasonSummary(player, position)}
                  </p>
                </div>
              </>
            ) : (
              <div className="empty-state">Search for a player to begin.</div>
            )}
          </article>
          <MatchupCard player={player} matchup={matchup} />
        </div>
        {error && <div className="error-banner">{error}</div>}
      </section>
      {prediction ? (
        <PredictionOutput prediction={prediction} />
      ) : positionPrediction ? (
        <PositionPredictionOutput prediction={positionPrediction} />
      ) : (
        <section className="preflight">
          <Database size={22} />
          <div>
            <strong>{STATIC_DATA ? loading ? "Loading published forecast" : "Select a published forecast" : "Ready for inference"}</strong>
            <span>
              {STATIC_DATA ? "Forecasts are computed in advance. No models run in response to your selection." : "Player history, snap usage, schedule context, and opponent WR allowance will be assembled when you run the models."}
            </span>
          </div>
        </section>
      )}
    </main>
  );
}
function PlayersPage({ onOpenForecast }: { onOpenForecast: OpenForecast }) {
  const [players, setPlayers] = useState<Player[]>([]);
  const [position, setPosition] = useState<Position>("WR");
  const [sortKey, setSortKey] = useState("default");
  useEffect(() => {
    api
      .players("", 2026, 40, position)
      .then(setPlayers)
      .catch(() => setPlayers([]));
  }, [position]);
  const perGame = (value: number | undefined, player: Player) => (value ?? 0) / Math.max(1, player.games_played);
  const columnGroups: Record<Position, Array<{ key: string; label: string; value: (player: Player) => number; format?: (value: number) => string }>> = {
    WR: [
      { key: "yards", label: "Receiving yards", value: (player) => player.season_receiving_yards },
      { key: "ypg", label: "Yards / game", value: (player) => perGame(player.season_receiving_yards, player), format: (value) => value.toFixed(1) },
      { key: "receptions", label: "Receptions", value: (player) => player.season_receptions ?? 0 },
      { key: "rpg", label: "Rec / game", value: (player) => perGame(player.season_receptions, player), format: (value) => value.toFixed(1) },
      { key: "pprpg", label: "PPR / game", value: (player) => perGame(player.season_ppr_points, player), format: (value) => value.toFixed(1) },
    ],
    QB: [
      { key: "passing", label: "Passing yards", value: (player) => player.season_passing_yards ?? 0 },
      { key: "rushing", label: "Rush yards", value: (player) => player.season_rushing_yards ?? 0 },
      { key: "ypg", label: "Pass yards / game", value: (player) => perGame(player.season_passing_yards, player), format: (value) => value.toFixed(1) },
      { key: "pprpg", label: "Fantasy / game", value: (player) => perGame(player.season_ppr_points, player), format: (value) => value.toFixed(1) },
    ],
    RB: [
      { key: "carries", label: "Carries", value: (player) => player.season_carries ?? 0 },
      { key: "rush", label: "Rush yards", value: (player) => player.season_rushing_yards ?? 0 },
      { key: "receptions", label: "Receptions", value: (player) => player.season_receptions ?? 0 },
      { key: "rpg", label: "Rec / game", value: (player) => perGame(player.season_receptions, player), format: (value) => value.toFixed(1) },
      { key: "recYards", label: "Rec yards", value: (player) => player.season_receiving_yards },
      { key: "pprpg", label: "PPR / game", value: (player) => perGame(player.season_ppr_points, player), format: (value) => value.toFixed(1) },
    ],
    TE: [
      { key: "yards", label: "Receiving yards", value: (player) => player.season_receiving_yards },
      { key: "ypg", label: "Yards / game", value: (player) => perGame(player.season_receiving_yards, player), format: (value) => value.toFixed(1) },
      { key: "receptions", label: "Receptions", value: (player) => player.season_receptions ?? 0 },
      { key: "rpg", label: "Rec / game", value: (player) => perGame(player.season_receptions, player), format: (value) => value.toFixed(1) },
      { key: "pprpg", label: "PPR / game", value: (player) => perGame(player.season_ppr_points, player), format: (value) => value.toFixed(1) },
    ],
  };
  const columns = columnGroups[position];
  const activeSort = sortKey === "default" ? columns[0].key : sortKey;
  const sortedPlayers = [...players].sort((left, right) => {
    const column = columns.find((item) => item.key === activeSort) ?? columns[0];
    return column.value(right) - column.value(left);
  });
  const updatePosition = (nextPosition: Position) => {
    setPosition(nextPosition);
    setSortKey("default");
  };
  return (
    <main className="page-content">
      <section className="page-intro">
        <div>
          <span className="eyebrow">2026 receiver pool</span>
          <h1>{position} leaderboard</h1>
          <p>Current-season production and availability from nflverse.</p>
          <div className="leaderboard-position-select">
            <span>Position</span>
            <PositionTabs value={position} onChange={updatePosition} />
          </div>
        </div>
      </section>
      <div className="table-shell">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Player</th>
              <th>Team</th>
              <th>Games</th>
              {columns.map((column) => (
                <th key={column.key}>
                  <button className="table-sort" onClick={() => setSortKey(column.key)}>
                    {column.label}{activeSort === column.key ? " ↓" : ""}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedPlayers.map((player, index) => (
              <tr key={player.player_id}>
                <td className="rank-cell">{index + 1}</td>
                <td>
                  <div className="table-player">
                    <PlayerPortrait player={player} />
                    <button className="player-forecast-link" onClick={() => onOpenForecast(player.player_id, position)} aria-label={`View ${player.name}'s forecast`}>{player.name}</button>
                  </div>
                </td>
                <td>
                  <TeamMark team={player.team} size="sm" />
                </td>
                <td>{player.games_played}</td>
                {columns.map((column) => {
                  const value = column.value(player);
                  return (
                    <td className={activeSort === column.key ? "sorted-cell" : ""} key={column.key}>
                      {column.format ? column.format(value) : value.toFixed(0)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </main>
  );
}
type ROSSort = "projected_yards" | "projected_passing_tds" | "projected_rushing_yards" | "projected_receiving_yards" | "projected_receptions" | "projected_rushing_tds" | "projected_receiving_tds" | "projected_touchdowns" | "projected_ppr_points";
function ROSRankingsPage({ currentWeek, availableWeeks, onOpenForecast }: { currentWeek: number; availableWeeks?: number[]; onOpenForecast: OpenForecast }) {
  const [rankings, setRankings] = useState<Array<ROSRanking | WeeklyRanking>>([]);
  const [position, setPosition] = useState<Position>("WR");
  const [mode, setMode] = useState<"weekly" | "ros">("weekly");
  const [rankingWeek, setRankingWeek] = useState(currentWeek);
  const [sortBy, setSortBy] = useState<ROSSort>("projected_ppr_points");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [injuries, setInjuries] = useState<Record<string, InjuryTag>>({});
  useEffect(() => setRankingWeek(currentWeek), [currentWeek]);
  useEffect(() => {
    let active = true;
    api
      .injuries(2026, mode === "weekly" ? rankingWeek : currentWeek)
      .then((tags) => active && setInjuries(tags))
      .catch(() => active && setInjuries({}));
    return () => {
      active = false;
    };
  }, [mode, rankingWeek, currentWeek]);
  useEffect(() => {
    let active = true;
    const request = mode === "weekly"
      ? api.weeklyRankings(2026, rankingWeek, currentWeek, position)
      : api.rosRankings(2026, currentWeek, position);

    request
      .then((rows) => {
        if (!active) return;
        setRankings(rows);
      })
      .catch((cause) => {
        if (!active) return;
        setError(cause instanceof Error ? cause.message : "Rankings unavailable");
      })
      .finally(() => {
        if (!active) return;
        setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [mode, position, rankingWeek, currentWeek]);
  const sortValue = (row: ROSRanking | WeeklyRanking, key: ROSSort) => {
    if (key === "projected_passing_tds") return row.predictions?.passing_tds ?? 0;
    if (key === "projected_rushing_yards") return row.predictions?.rushing_yards ?? 0;
    if (key === "projected_receiving_yards") return row.predictions?.receiving_yards ?? 0;
    if (key === "projected_rushing_tds") return row.predictions?.rushing_tds ?? 0;
    if (key === "projected_receiving_tds") return row.predictions?.receiving_tds ?? 0;
    if (key === "projected_yards") return row.projected_yards;
    if (key === "projected_receptions") return row.projected_receptions;
    if (key === "projected_touchdowns") return row.projected_touchdowns;
    return row.projected_ppr_points;
  };
  const sorted = [...rankings].sort((left, right) => sortValue(right, sortBy) - sortValue(left, sortBy));
  const rankingYardsLabel = position === "QB" ? "Pass yards" : position === "RB" ? "Total yards" : "Rec yards";
  const rosYardsLabel = position === "QB" ? "ROS pass yards" : position === "RB" ? "ROS total yards" : "ROS yards";
  const sortHeader = (id: ROSSort, label: string) => (
    <button className="table-sort" onClick={() => setSortBy(id)}>
      {label}{sortBy === id ? " ↓" : ""}
    </button>
  );
  const changeMode = (nextMode: "weekly" | "ros") => {
    setLoading(true);
    setError("");
    setRankings([]);
    setMode(nextMode);
  };
  const changePosition = (nextPosition: Position) => {
    setLoading(true);
    setError("");
    setRankings([]);
    setPosition(nextPosition);
  };
  const changeWeek = (nextWeek: number) => {
    setLoading(true);
    setError("");
    setRankings([]);
    setRankingWeek(nextWeek);
  };
  return (
    <main className="page-content">
      <section className="page-intro rankings-intro">
        <div>
          <span className="eyebrow">
            {mode === "weekly" ? `Week ${rankingWeek}` : `Weeks ${currentWeek}–18`} · standard PPR
          </span>
          <h1>{position} rankings</h1>
          <p>
            {mode === "weekly"
              ? "Matchup-specific receiving and fantasy projections."
              : "Current roles and efficiency projected across each team’s remaining schedule."}
          </p>
        </div>
        <div className="ranking-tools">
          <div className="ranking-tool-group">
            <span>View</span>
            <div className="mode-controls" aria-label="Ranking period">
              <button className={mode === "weekly" ? "active" : ""} onClick={() => changeMode("weekly")}>Weekly</button>
              <button className={mode === "ros" ? "active" : ""} onClick={() => changeMode("ros")}>ROS</button>
            </div>
          </div>
          <div className="ranking-tool-group">
            <span>Position</span>
            <PositionTabs value={position} onChange={changePosition} />
          </div>
          {mode === "weekly" && (
            <div className="ranking-tool-group ranking-week">
              <span>Week</span>
              <CompactSelect
                label="Ranking week"
                value={rankingWeek}
                options={STATIC_DATA ? availableWeeks ?? [currentWeek] : Array.from(
                  { length: Math.max(0, 19 - currentWeek) },
                  (_, index) => index + currentWeek,
                )}
                onChange={changeWeek}
              />
            </div>
          )}
        </div>
      </section>
      {loading && <div className="empty-state tall">Loading {mode === "weekly" ? `Week ${rankingWeek}` : "rest-of-season"} projections…</div>}
      {error && <div className="error-banner">{error}</div>}
      {!loading && !error && (
        <div className="table-shell ros-table">
          <table>
            <thead>
              <tr>
                <th>#</th><th>Player</th><th>Team</th>
                {mode === "weekly" ? <th>Opponent</th> : <th>Games left</th>}
                {position === "QB" ? (
                  <>
                    <th className={sortBy === "projected_yards" ? "sorted-header" : ""}>{sortHeader("projected_yards", mode === "weekly" ? "Pass yards" : "ROS pass yards")}</th>
                    <th className={sortBy === "projected_passing_tds" ? "sorted-header" : ""}>{sortHeader("projected_passing_tds", mode === "weekly" ? "Pass TDs" : "ROS pass TDs")}</th>
                    <th className={sortBy === "projected_rushing_yards" ? "sorted-header" : ""}>{sortHeader("projected_rushing_yards", mode === "weekly" ? "Rush yards" : "ROS rush yards")}</th>
                  </>
                ) : position === "RB" ? (
                  <>
                    <th className={sortBy === "projected_rushing_yards" ? "sorted-header" : ""}>{sortHeader("projected_rushing_yards", mode === "weekly" ? "Rush yards" : "ROS rush yards")}</th>
                    <th className={sortBy === "projected_receiving_yards" ? "sorted-header" : ""}>{sortHeader("projected_receiving_yards", mode === "weekly" ? "Rec yards" : "ROS rec yards")}</th>
                    <th className={sortBy === "projected_receptions" ? "sorted-header" : ""}>{sortHeader("projected_receptions", mode === "weekly" ? "Receptions" : "ROS receptions")}</th>
                    <th className={sortBy === "projected_rushing_tds" ? "sorted-header" : ""}>{sortHeader("projected_rushing_tds", mode === "weekly" ? "Rush TDs" : "ROS rush TDs")}</th>
                    <th className={sortBy === "projected_receiving_tds" ? "sorted-header" : ""}>{sortHeader("projected_receiving_tds", mode === "weekly" ? "Rec TDs" : "ROS rec TDs")}</th>
                  </>
                ) : (
                  <>
                    <th className={sortBy === "projected_yards" ? "sorted-header" : ""}>{sortHeader("projected_yards", mode === "weekly" ? rankingYardsLabel : rosYardsLabel)}</th>
                    <th className={sortBy === "projected_receptions" ? "sorted-header" : ""}>{sortHeader("projected_receptions", mode === "weekly" ? "Receptions" : "ROS receptions")}</th>
                    <th className={sortBy === "projected_touchdowns" ? "sorted-header" : ""}>{sortHeader("projected_touchdowns", mode === "weekly" ? "Rec TDs" : "ROS TDs")}</th>
                  </>
                )}
                <th className={sortBy === "projected_ppr_points" ? "sorted-header" : ""}>{sortHeader("projected_ppr_points", mode === "weekly" ? "PPR points" : "ROS PPR")}</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((player, index) => (
                <tr key={player.player_id}>
                  <td className="rank-cell">{index + 1}</td>
                  <td>
                    <div className="table-player">
                      <button className="player-forecast-link" onClick={() => onOpenForecast(player.player_id, position, mode === "weekly" ? rankingWeek : currentWeek)} aria-label={`View ${player.player_name}'s forecast`}>{player.player_name}</button>
                      {injuries[player.player_id] && (
                        <span
                          className={`health-tag health-${injuries[player.player_id].risk}`}
                          title={`${injuries[player.player_id].injury ?? injuries[player.player_id].label}${
                            injuries[player.player_id].play_probability < 1
                              ? ` · ${Math.round(injuries[player.player_id].play_probability * 100)}% to play, projection adjusted`
                              : ""
                          }`}
                        >
                          {injuries[player.player_id].label}
                          {injuries[player.player_id].play_probability < 1 && mode === "weekly"
                            ? ` ${Math.round(injuries[player.player_id].play_probability * 100)}%`
                            : ""}
                        </span>
                      )}
                    </div>
                  </td>
                  <td><TeamMark team={player.team} size="sm" /></td>
                  <td>{mode === "weekly" ? (player as WeeklyRanking).opponent : (player as ROSRanking).remaining_games}</td>
                  {position === "QB" ? (
                    <>
                      <td className={sortBy === "projected_yards" ? "sorted-cell" : ""}>{player.projected_yards.toFixed(1)}</td>
                      <td className={sortBy === "projected_passing_tds" ? "sorted-cell" : ""}>{(player.predictions?.passing_tds ?? 0).toFixed(1)}</td>
                      <td className={sortBy === "projected_rushing_yards" ? "sorted-cell" : ""}>{(player.predictions?.rushing_yards ?? 0).toFixed(1)}</td>
                    </>
                  ) : position === "RB" ? (
                    <>
                      <td className={sortBy === "projected_rushing_yards" ? "sorted-cell" : ""}>{(player.predictions?.rushing_yards ?? 0).toFixed(1)}</td>
                      <td className={sortBy === "projected_receiving_yards" ? "sorted-cell" : ""}>{(player.predictions?.receiving_yards ?? 0).toFixed(1)}</td>
                      <td className={sortBy === "projected_receptions" ? "sorted-cell" : ""}>{player.projected_receptions.toFixed(1)}</td>
                      <td className={sortBy === "projected_rushing_tds" ? "sorted-cell" : ""}>{(player.predictions?.rushing_tds ?? 0).toFixed(1)}</td>
                      <td className={sortBy === "projected_receiving_tds" ? "sorted-cell" : ""}>{(player.predictions?.receiving_tds ?? 0).toFixed(1)}</td>
                    </>
                  ) : (
                    <>
                      <td className={sortBy === "projected_yards" ? "sorted-cell" : ""}>{player.projected_yards.toFixed(1)}</td>
                      <td className={sortBy === "projected_receptions" ? "sorted-cell" : ""}>{player.projected_receptions.toFixed(1)}</td>
                      <td className={sortBy === "projected_touchdowns" ? "sorted-cell" : ""}>{player.projected_touchdowns.toFixed(1)}</td>
                    </>
                  )}
                  <td className={sortBy === "projected_ppr_points" ? "sorted-cell" : ""}>{player.projected_ppr_points.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}
function HistoryPage({ refreshKey }: { refreshKey: number }) {
  const [history, setHistory] = useState<PredictionHistory[]>([]);
  useEffect(() => {
    api
      .history()
      .then(setHistory)
      .catch(() => setHistory([]));
  }, [refreshKey]);
  return (
    <main className="page-content">
      <section className="page-intro">
        <div>
          <span className="eyebrow">Saved model runs</span>
          <h1>Prediction history</h1>
          <p>
            Auditable snapshots of inputs and outputs generated by the
            production models.
          </p>
        </div>
      </section>
      <div className="history-list">
        {history.map((item) => {
          const primary = item.ensemble_prediction ?? item.predictions?.numerical_mlp;
          return (
            <article key={item.id} className="history-row">
              <div className="history-date">
                <span>{formatDate(item.game_date)}</span>
                <small>{item.game_id}</small>
              </div>
              <div className="history-player">
                <strong>{item.player_name}</strong>
                <span>
                  <TeamMark team={item.team} size="sm" /> {item.team} at{" "}
                  {item.opponent}
                </span>
              </div>
              <div className="history-models">
                <span>RIDGE {item.predictions?.ridge?.toFixed(1) ?? "—"}</span>
                <span>XGB {item.predictions?.xgboost?.toFixed(1) ?? "—"}</span>
              </div>
              <div className="history-primary">
                <strong>{primary?.toFixed(1) ?? "—"}</strong>
                <span>{item.ensemble_prediction == null ? "MLP YDS" : "BLEND YDS"}</span>
              </div>
            </article>
          );
        })}
        {history.length === 0 && (
          <div className="empty-state tall">No saved predictions yet.</div>
        )}
      </div>
    </main>
  );
}
function ModelsPage() {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  useEffect(() => {
    api
      .metrics()
      .then(setMetrics)
      .catch(() => setMetrics(null));
  }, []);
  const cohort = metrics?.base.common_test_cohort ?? {};
  return (
    <main className="page-content">
      <section className="page-intro">
        <div>
          <span className="eyebrow">Evaluation registry</span>
          <h1>Model performance</h1>
          <p>Historical WR receiving-yards benchmarks on the 2025 holdout. Updated after retraining and evaluation, not live data refreshes.</p>
        </div>
      </section>
      <div className="metric-banner">
        <div>
          <span>Production model</span>
          <strong>Validation-weighted ensemble</strong>
        </div>
        <div>
          <span>Training window</span>
          <strong>2016–2023</strong>
        </div>
        <div>
          <span>Holdout</span>
          <strong>2025 season</strong>
        </div>
      </div>
      <div className="model-table-grid">
        {Object.entries(cohort).map(([name, values]) => (
          <article
            className={`metric-card ${name === metrics?.base.lowest_test_mae ? "best" : ""}`}
            key={name}
          >
            <div className="metric-card-title">
              <span>{modelLabel(name)}</span>
              {name === metrics?.base.lowest_test_mae && (
                <span className="primary-tag">Lowest MAE</span>
              )}
            </div>
            <strong>{values.mae.toFixed(2)}</strong>
            <small>Mean absolute error</small>
            <dl>
              <div>
                <dt>RMSE</dt>
                <dd>{values.rmse.toFixed(2)}</dd>
              </div>
              <div>
                <dt>Median AE</dt>
                <dd>{values.median_absolute_error.toFixed(2)}</dd>
              </div>
              <div>
                <dt>Rows</dt>
                <dd>{values.rows.toLocaleString()}</dd>
              </div>
            </dl>
          </article>
        ))}
      </div>
    </main>
  );
}

function App() {
  const [forecastSelection, setForecastSelection] = useState<ForecastSelection | null>(null);
  const openForecast: OpenForecast = (playerId, position, week) => {
    setForecastSelection({ playerId, position, week: week ?? health?.current_week ?? 4 });
    setPage("prediction");
    setMobileNav(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  const [page, setPage] = useState<Page>("prediction"),
    [mobileNav, setMobileNav] = useState(false),
    [health, setHealth] = useState<ApiHealth | null>(null),
    [historyRefresh, setHistoryRefresh] = useState(0);
  useEffect(() => {
    let active = true;
    const refreshHealth = () => {
      api
        .health()
        .then((nextHealth) => {
          if (active) setHealth(nextHealth);
        })
        .catch(() => {
          if (active) setHealth(null);
        });
    };
    refreshHealth();
    const interval = window.setInterval(refreshHealth, 60_000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);
  const navigate = (next: Page) => {
    if (next === "ros") {
      void Promise.all([
        api.weeklyRankings(2026, health?.current_week ?? 4, health?.current_week ?? 4),
        api.rosRankings(2026, health?.current_week ?? 4),
      ]).catch(() => undefined);
    }
    startTransition(() => setPage(next));
    setMobileNav(false);
  };
  return (
    <div className="app-shell">
      <header className="topbar">
        <button
          className="mobile-menu icon-button"
          onClick={() => setMobileNav((open) => !open)}
          aria-label={mobileNav ? "Close navigation" : "Open navigation"}
        >
          <Menu />
        </button>
        <button className="brand" onClick={() => navigate("prediction")}>
          <span className="brand-mark">TS</span>
          <span>
            <b>THE SCRIPT</b>
            <small>NFL FORECAST LAB</small>
          </span>
        </button>
        <nav className="top-nav" aria-label="Primary navigation">
          {NAV_ITEMS.map(({ id, label }) => (
            <button
              key={id}
              className={page === id ? "active" : ""}
              onClick={() => navigate(id)}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>
      <aside className={`sidebar ${mobileNav ? "open" : ""}`}>
        <div className="sidebar-head">
          <span>Workspace</span>
          <button
            className="icon-button sidebar-menu-toggle"
            onClick={() => setMobileNav(false)}
            aria-label="Close navigation"
          >
            <Menu size={22} />
          </button>
        </div>
        <nav>
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={page === id ? "active" : ""}
              onClick={() => navigate(id)}
            >
              <Icon size={17} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-meta">
          <span>Data cutoff</span>
          <strong>Week {health?.latest_completed_week ?? "—"}, 2026</strong>
          <small>Prediction contract: 24 hours before kickoff</small>
        </div>
      </aside>
      {mobileNav && (
        <button
          className="nav-scrim"
          onClick={() => setMobileNav(false)}
          aria-label="Close navigation overlay"
        />
      )}
      <div className="app-main">
        {page === "prediction" && (
          <PredictionPage
            key={forecastSelection ? `${forecastSelection.playerId}-${forecastSelection.week}` : "default"}
            selection={forecastSelection}
            currentWeek={health?.current_week ?? 4}
            health={health}
            onSaved={() => setHistoryRefresh((value) => value + 1)}
          />
        )}
        {page === "players" && <PlayersPage onOpenForecast={openForecast} />}
        {page === "ros" && <ROSRankingsPage currentWeek={health?.current_week ?? 4} availableWeeks={health?.available_weeks} onOpenForecast={openForecast} />}
        {page === "history" && <HistoryPage refreshKey={historyRefresh} />}
        {page === "models" && <ModelsPage />}
      </div>
    </div>
  );
}
function formatDate(value: string) {
  const date = new Date(`${value}T12:00:00`);
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(date);
}
function formatNumber(value: number | null | undefined) {
  return value == null ? "—" : value.toFixed(1);
}
function formatPercent(value: number | null | undefined) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}
function formatDepthRole(value: number | null | undefined) {
  return value == null ? "—" : `WR${Math.round(value)}`;
}
function modelLabel(value: string) {
  if (value === "xgboost") return "XGBoost";
  if (value === "xgboost_poisson") return "XGBoost Poisson";
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}
export default App;
