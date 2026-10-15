import type { Player, PredictionHistory } from './api'

export const STATIC_DATA = import.meta.env.VITE_DATA_MODE === 'static'
const BASE = (import.meta.env.VITE_DATA_URL ?? '/data').replace(/\/$/, '')
type Manifest = {
  release_id: string; season: number; current_week: number; available_weeks: number[];
  generated_at: string; data_through_week: number; preview_only: boolean;
  source_freshness: { stale: boolean };
  files: { players: Record<string, string>; forecasts: Record<string, Record<string, string>>;
    matchups: Record<string, Record<string, string>>; weekly: Record<string, Record<string, string>>;
    ros: Record<string, string>; injuries: Record<string, string>; profiles: Record<string, string>;
    health: string; metrics: string };
}
let discovery: { expires: number; promise: Promise<Manifest> } | undefined
const files = new Map<string, Promise<unknown>>()
async function read<T>(path: string, fresh = false): Promise<T> {
  if (!path) throw new Error('This forecast is not available in the published release.')
  if (!fresh && files.has(path)) return files.get(path) as Promise<T>
  const pending = fetch(`${BASE}/${path}`, { cache: fresh ? 'no-store' : 'default' }).then(async response => {
    if (!response.ok) throw new Error('Published forecast data is unavailable. Please try again later.')
    return response.json() as Promise<T>
  }).catch(error => { files.delete(path); throw error })
  if (!fresh) files.set(path, pending)
  return pending
}
export function manifest(): Promise<Manifest> {
  if (!discovery || discovery.expires < Date.now()) {
    discovery = { expires: Date.now() + 60_000, promise: read<Manifest>('manifest.json', true) }
    discovery.promise.catch(() => { discovery = undefined })
  }
  return discovery.promise
}
function history(): PredictionHistory[] {
  try { return JSON.parse(localStorage.getItem('the-script-history') ?? '[]') } catch { return [] }
}
export async function staticRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const url = new URL(path, 'https://static.invalid')
  const route = url.pathname
  if (route === '/predictions') return history() as T
  const m = await manifest()
  const season = Number(url.searchParams.get('season') ?? m.season)
  if (season !== m.season) throw new Error(`Published forecasts cover the ${m.season} season only.`)
  const week = String(url.searchParams.get('week') ?? m.current_week)
  const position = url.searchParams.get('position') ?? 'WR'
  if (route === '/health') return { ...await read<object>(m.files.health), generated_at: m.generated_at,
    available_weeks: m.available_weeks, source_stale: m.source_freshness.stale, preview_only: m.preview_only } as T
  if (route === '/models/metrics') return read<T>(m.files.metrics)
  if (route === '/injuries') return read<T>(m.files.injuries[week])
  if (route === '/position-ros-rankings') return read<T>(m.files.ros[position])
  if (route === '/weekly-rankings') return read<T>(m.files.weekly[position]?.[week])
  if (route === '/players') {
    const players = await read<Player[]>(m.files.players[position])
    const query = (url.searchParams.get('query') ?? '').trim().toLowerCase()
    return players.filter(p => p.name.toLowerCase().includes(query)).slice(0, Number(url.searchParams.get('limit') ?? 12)) as T
  }
  if (route.startsWith('/players/')) return read<T>(m.files.profiles[route.split('/')[2]])
  if (route.startsWith('/matchup/')) {
    const id = route.split('/')[2]
    for (const pos of Object.keys(m.files.players)) {
      const players = await read<Player[]>(m.files.players[pos])
      if (players.some(p => p.player_id === id)) {
        const matchups = await read<Record<string, T>>(m.files.matchups[pos]?.[week])
        if (!matchups[id]) throw new Error('No scheduled matchup for this player in this week.')
        return matchups[id]
      }
    }
  }
  if (route === '/predict' || route === '/predict-position') {
    const body = JSON.parse(String(init?.body))
    if (body.season !== m.season) throw new Error('That season is not published.')
    const pos = body.position ?? 'WR'
    const players = await read<Player[]>(m.files.players[pos])
    const player = players.find(p => p.name === body.player_name && p.team === body.team)
    const forecasts = await read<Record<string, T>>(m.files.forecasts[pos]?.[String(body.week)])
    if (!player || !forecasts[player.player_id]) throw new Error('No published forecast for this player and week.')
    const forecast = forecasts[player.player_id]
    if (pos === 'WR') {
      const item = { ...forecast as object, id: `${player.player_id}_${body.week}`, created_at: new Date().toISOString(), file_name: '' } as PredictionHistory
      try { localStorage.setItem('the-script-history', JSON.stringify([item, ...history().filter(h => h.id !== item.id)].slice(0, 20))) } catch { /* Browsing remains available when storage is full. */ }
    }
    return forecast
  }
  throw new Error('This action is not supported in static forecast mode.')
}