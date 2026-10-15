import { STATIC_DATA, staticRequest } from './staticData'
export { STATIC_DATA } from './staticData'
const API_BASE = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000/api'

export type ApiHealth = { status: string; service?: string; season?: number; current_week?: number; latest_completed_week?: number | null; models_ready: boolean; checked_at?: string; generated_at?: string; available_weeks?: number[]; source_stale?: boolean; preview_only?: boolean }
export type Position = 'QB' | 'RB' | 'WR' | 'TE'
export type Player = { player_id: string; name: string; team: string; position?: string; headshot_url?: string | null; games_played: number; latest_week?: number; season_receiving_yards: number; season_receiving_tds?: number; season_passing_yards?: number; season_rushing_yards?: number; season_rushing_tds?: number; season_carries?: number; season_ppr_points?: number; season_targets?: number; season_receptions?: number }
export type Matchup = { game_id: string; season: number; week: number; game_date: string; game_time?: string | null; team: string; opponent: string; home_game: boolean; days_rest: number; stadium?: string | null; roof?: string | null; temperature?: number | null; wind?: number | null }
export type RecentGame = { week: number; game_id: string; opponent: string; yards: number; targets: number; receptions: number; touchdowns: number }
export type Prediction = {
  player_name: string; player_id: string; game_id: string; game_date: string; team: string; opponent: string;
  prediction_time_contract: string; data_cutoff?: string; primary_model: string; predicted_yards: number;
  model_range: { low: number; high: number }; recent_games: RecentGame[];
  predictions: { ridge: number; xgboost: number; numerical_mlp: number };
  volume_predictions: {
    targets: { ridge: number; xgboost: number };
    receptions: { ridge: number; xgboost: number };
  };
  expected_targets: number;
  expected_receptions: number;
  touchdown_predictions: { poisson: number; xgboost_poisson: number };
  touchdown_model: string;
  expected_receiving_touchdowns: number;
  anytime_touchdown_probability: number;
  receiving_yards_quantiles: { q10: number; q50: number; q90: number };
  decomposition_prediction: number;
  ensemble_weights: Record<string, number>;
  ensemble_prediction: number;
  features: Record<string, number | null>;
  opponent_context: { yards_allowed_last_game?: number | null; yards_allowed_avg_last_5?: number | null; ppr_points_allowed_avg_last_5?: number | null };
  generated_at: string;
}
export type PredictionHistory = Partial<Prediction> & { id: string; file_name: string; created_at: string; player_name: string; game_id: string; game_date: string; team: string; opponent: string; predictions?: Prediction['predictions'] }
export type InjuryTag = { risk: 'high' | 'medium'; label: string; play_probability: number; injury?: string | null }
export type PositionRankingStats = { passing_yards?: number; passing_tds?: number; rushing_yards?: number; rushing_tds?: number; carries?: number; receiving_yards?: number; receptions?: number; receiving_tds?: number }
export type ROSRanking = { player_id: string; player_name: string; team: string; position?: Position; remaining_games: number; projected_yards: number; projected_receptions: number; projected_touchdowns: number; projected_ppr_points: number; unadjusted_ppr_points?: number; play_probability?: number; predictions?: PositionRankingStats }
export type WeeklyRanking = Omit<ROSRanking, 'remaining_games'> & { week: number; opponent: string }
export type PositionRecentGame = { week: number; game_id: string; opponent: string; value: number; secondary_value?: number | null }
export type PositionPrediction = { player_name: string; player_id: string; position: Position; team: string; opponent: string; game_id: string; game_date: string; predictions: Record<string, number>; selected_models: Record<string, string>; model_predictions?: Record<string, number>; features: Record<string, number | null>; recent_games: PositionRecentGame[]; secondary_recent_label?: string | null; primary_target: string; primary_quantiles: { q10: number; q50: number; q90: number }; projected_ppr_points: number }
type MetricValues = { rows: number; mae: number; rmse: number; median_absolute_error: number }
export type ModelMetrics = { base: { common_test_cohort: Record<string, MetricValues>; lowest_test_mae: string }; opponent_experiment: unknown; production_model: string; training_seasons: string; validation_season: number; test_season: number }

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (STATIC_DATA) return staticRequest<T>(path, init)
  const response = await fetch(`${API_BASE}${path}`, { ...init, headers: { 'Content-Type': 'application/json', ...init?.headers } })
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: 'Request failed' }))
    throw new Error(body.detail ?? `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

const rankingCache = new Map<string, { expiresAt: number; pending: Promise<unknown> }>()

function cachedRequest<T>(path: string): Promise<T> {
  const existing = rankingCache.get(path)
  if (existing && existing.expiresAt > Date.now()) return existing.pending as Promise<T>
  const pending = request<T>(path).catch((error) => {
    rankingCache.delete(path)
    throw error
  })
  rankingCache.set(path, { pending, expiresAt: Date.now() + 60_000 })
  return pending
}

export const api = {
  health: () => request<ApiHealth>('/health?season=2026'),
  players: (query = '', season = 2026, limit = 12, position: Position = 'WR') => request<Player[]>(`/players?query=${encodeURIComponent(query)}&season=${season}&limit=${limit}&position=${position}`),
  player: (id: string, season = 2026) => request<Player & { recent_games: RecentGame[] }>(`/players/${id}?season=${season}`),
  matchup: (id: string, season: number, week: number) => request<Matchup>(`/matchup/${id}?season=${season}&week=${week}`),
  predict: (body: { player_name: string; team: string; season: number; week: number; exclude_current_season_data: boolean }) => request<Prediction>('/predict', { method: 'POST', body: JSON.stringify(body) }),
  predictPosition: (body: { player_name: string; position: Position; team: string; season: number; week: number }) => request<PositionPrediction>('/predict-position', { method: 'POST', body: JSON.stringify(body) }),
  history: () => request<PredictionHistory[]>('/predictions'),
  rosRankings: (season = 2026, week = 4, position: Position = 'WR') => cachedRequest<ROSRanking[]>(`/position-ros-rankings?season=${season}&week=${week}&position=${position}`),
  weeklyRankings: (season = 2026, week = 4, dataCutoffWeek = 4, position: Position = 'WR') => cachedRequest<WeeklyRanking[]>(`/weekly-rankings?season=${season}&week=${week}&data_cutoff_week=${dataCutoffWeek}&position=${position}`),
  prefetchRankings: () => Promise.all([
    api.weeklyRankings(),
    api.rosRankings(),
  ]).then(() => undefined),
  injuries: (season = 2026, week?: number) => request<Record<string, InjuryTag>>(`/injuries?season=${season}${week ? `&week=${week}` : ''}`),
  metrics: () => request<ModelMetrics>('/models/metrics'),
}
