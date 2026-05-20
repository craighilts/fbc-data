#!/usr/bin/env node
/**
 * FBC GHIN Handicap Fetcher (v5 — Excel-first architecture)
 *
 * Data flow:
 *   1. Roster + static historical data (draft_hi, fbc12_index, played_fbc12)
 *      come from data/members.json — never overwritten by GHIN.
 *   2. Current HI + 52-week low come from:
 *      a) Live GHIN if a real GHIN number is configured for that member, OR
 *      b) Seeded values from members.json (Excel-sourced) otherwise.
 *   3. Manual-handicap players (Norat, Connors) always use the values in their
 *      manual_handicap block.
 *   4. Course handicaps are computed from current HI for both Frisco courses.
 *   5. GHIN players also get last 20 scores → form analysis + sparkline data.
 *
 * Page will work on day 1 with Excel data for everyone. Adding GHIN numbers
 * over time progressively upgrades each player to live data + sparklines.
 *
 * Required environment variables (only if any member has a real GHIN number):
 *   GHIN_EMAIL    — your GHIN.com account email
 *   GHIN_PASSWORD — your GHIN.com account password
 */

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------- Configuration ----------
const GHIN_API_BASE = 'https://api2.ghin.com/api/v1';
const MEMBERS_FILE = path.resolve(__dirname, '..', 'data', 'members.json');
const COURSES_FILE = path.resolve(__dirname, '..', 'data', 'courses.json');
const OUTPUT_FILE = path.resolve(__dirname, '..', 'data', 'handicaps.json');
const PREVIOUS_FILE = OUTPUT_FILE;
const SCORES_LIMIT = 20;
const GHIN_PLACEHOLDER = 'PLACEHOLDER_ADD_GHIN_NUMBER';

const USER_AGENT = 'GHINcom/280 CFNetwork/1494.0.7 Darwin/23.4.0';

// ---------- Helpers ----------
function log(level, msg, extra) {
  const ts = new Date().toISOString();
  const line = `[${ts}] [${level}] ${msg}`;
  if (extra !== undefined) {
    console.log(line, typeof extra === 'string' ? extra : JSON.stringify(extra));
  } else {
    console.log(line);
  }
}

async function readJson(filePath) {
  const raw = await fs.readFile(filePath, 'utf8');
  return JSON.parse(raw);
}

async function readJsonIfExists(filePath) {
  try {
    return await readJson(filePath);
  } catch (err) {
    if (err.code === 'ENOENT') return null;
    throw err;
  }
}

async function writeJson(filePath, data) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, JSON.stringify(data, null, 2) + '\n', 'utf8');
}

// ---------- Roster Classification ----------
/**
 * Classify each member into one of three buckets:
 *   - 'ghin_live': has a real GHIN number (not the placeholder) — fetch from GHIN
 *   - 'ghin_seeded': has the placeholder GHIN — use seeded values from Excel
 *   - 'manual': has manual_handicap block — use those values
 */
function classifyMembers(rawMembers) {
  const liveGhinMembers = [];
  const seededMembers = [];
  const manualMembers = [];
  const errors = [];

  for (const m of rawMembers) {
    if (m._comment && !m.name) continue;
    if (!m.name) {
      errors.push(`Member missing 'name' field: ${JSON.stringify(m)}`);
      continue;
    }

    const hasGhin = m.ghin != null && m.ghin !== '';
    const hasManual = m.manual_handicap != null &&
      typeof m.manual_handicap === 'object' &&
      m.manual_handicap.handicap_index != null;

    if (hasGhin && hasManual) {
      errors.push(`Member "${m.name}" has both 'ghin' and 'manual_handicap'. Use one or the other.`);
      continue;
    }
    if (!hasGhin && !hasManual) {
      errors.push(`Member "${m.name}" has neither 'ghin' nor 'manual_handicap'.`);
      continue;
    }

    if (hasManual) {
      manualMembers.push(m);
    } else if (m.ghin === GHIN_PLACEHOLDER) {
      // Has the placeholder — use seeded data
      if (!m.seeded_handicap || m.seeded_handicap.handicap_index == null) {
        errors.push(`Member "${m.name}" has placeholder GHIN but no seeded_handicap data.`);
        continue;
      }
      seededMembers.push(m);
    } else {
      // Real GHIN number
      liveGhinMembers.push(m);
    }
  }

  return { liveGhinMembers, seededMembers, manualMembers, errors };
}

// ---------- Course Handicap Computation ----------
function computeCourseHandicap(handicapIndex, courseRating, slopeRating, par) {
  if (handicapIndex == null) return null;
  const raw = handicapIndex * (slopeRating / 113) + (courseRating - par);
  return Math.round(raw);
}

function computeCourseHandicaps(handicapIndex, courses) {
  const result = {};
  for (const [key, course] of Object.entries(courses)) {
    result[key] = {
      course_name: course.name,
      short_name: course.short_name,
      tee_name: course.tee_name,
      rating: course.rating,
      slope: course.slope,
      par: course.par,
      course_handicap: computeCourseHandicap(handicapIndex, course.rating, course.slope, course.par),
    };
  }
  return result;
}

// ---------- Historical Comparisons ----------
/**
 * Compute the three change metrics shown in the Excel:
 *   - Change from 52-week Low (regression from peak; positive = worse than peak)
 *   - Change from Draft (positive = HI has risen since draft, worse)
 *   - Change from FBC 12 (positive = HI has risen since FBC 12, worse)
 *
 * Convention matches the Excel: Change = Current - Reference. Positive means
 * the handicap went UP (player got worse). The UI will display directionally.
 */
function computeChanges(currentHi, member) {
  const changes = {
    vs_52wk_low: { change: null, pct_change: null },
    vs_draft: { change: null, pct_change: null },
    vs_fbc12: { change: null, pct_change: null },
  };

  if (currentHi == null) return changes;

  // Use the most recent low (from GHIN if available, else seeded)
  const lowHi = member._low_hi_resolved;
  if (lowHi != null) {
    const change = +(currentHi - lowHi).toFixed(2);
    const pct = lowHi !== 0 ? change / lowHi : null;
    changes.vs_52wk_low = {
      change,
      pct_change: pct != null ? +pct.toFixed(4) : null,
    };
  }

  if (member.draft_hi != null) {
    const change = +(currentHi - member.draft_hi).toFixed(2);
    const pct = member.draft_hi !== 0 ? change / member.draft_hi : null;
    changes.vs_draft = {
      change,
      pct_change: pct != null ? +pct.toFixed(4) : null,
    };
  }

  if (member.played_fbc12 && member.fbc12_index != null) {
    const change = +(currentHi - member.fbc12_index).toFixed(2);
    const pct = member.fbc12_index !== 0 ? change / member.fbc12_index : null;
    changes.vs_fbc12 = {
      change,
      pct_change: pct != null ? +pct.toFixed(4) : null,
    };
  }

  return changes;
}

// ---------- GHIN API ----------
async function ghinLogin(email, password) {
  const url = `${GHIN_API_BASE}/golfer_login.json`;
  const body = {
    user: { email_or_ghin: email, password, remember_me: false },
    token: 'nonce',
    source: 'GHINcom',
  };

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'User-Agent': USER_AGENT,
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`GHIN login failed: HTTP ${res.status} — ${text.slice(0, 200)}`);
  }

  const data = await res.json();
  const token = data?.golfer_user?.golfer_user_token;
  if (!token) throw new Error('GHIN login succeeded but no token in response');
  log('INFO', 'GHIN authentication successful');
  return token;
}

async function fetchGolfer(token, ghinNumber) {
  const url = `${GHIN_API_BASE}/golfers/search.json?golfer_id=${encodeURIComponent(ghinNumber)}&per_page=1&page=1`;
  const res = await fetch(url, {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Accept': 'application/json',
      'User-Agent': USER_AGENT,
    },
  });
  if (!res.ok) {
    log('WARN', `GHIN fetch failed for ${ghinNumber}: HTTP ${res.status}`);
    return null;
  }
  const data = await res.json();
  const golfer = Array.isArray(data?.golfers) ? data.golfers[0] : null;
  if (!golfer) {
    log('WARN', `No golfer found for GHIN ${ghinNumber}`);
    return null;
  }
  return golfer;
}

async function fetchScores(token, golferId, limit = SCORES_LIMIT) {
  const url = `${GHIN_API_BASE}/scores/search.json?golfer_id=${encodeURIComponent(golferId)}&per_page=${limit}&page=1`;
  const res = await fetch(url, {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Accept': 'application/json',
      'User-Agent': USER_AGENT,
    },
  });
  if (!res.ok) {
    log('WARN', `Scores fetch failed for ${golferId}: HTTP ${res.status}`);
    return [];
  }
  const data = await res.json();
  const scores = data?.scores ?? data?.Scores ?? [];
  return Array.isArray(scores) ? scores : [];
}

// ---------- Score Analysis ----------
function analyzeScores(scores, handicapIndex) {
  if (!Array.isArray(scores) || scores.length === 0) {
    return {
      score_count: 0,
      last_10_avg_differential: null,
      last_5_avg_differential: null,
      last_round_differential: null,
      rounds_last_30_days: 0,
      hot_or_cold_delta: null,
      form: 'unknown',
    };
  }

  const sorted = [...scores].sort((a, b) => {
    const da = new Date(a.played_at || a.posted_at || 0).getTime();
    const db = new Date(b.played_at || b.posted_at || 0).getTime();
    return db - da;
  });

  const diffs = sorted.map(s => parseFloat(s.differential)).filter(d => !isNaN(d));
  const avg = (arr) => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : null;

  const last10 = avg(diffs.slice(0, 10));
  const last5 = avg(diffs.slice(0, 5));
  const lastRound = diffs[0] ?? null;

  const thirtyDaysAgo = Date.now() - 30 * 24 * 60 * 60 * 1000;
  const recentCount = sorted.filter(s => {
    const t = new Date(s.played_at || s.posted_at || 0).getTime();
    return t > thirtyDaysAgo;
  }).length;

  let hotColdDelta = null;
  let form = 'steady';
  if (last5 != null && handicapIndex != null) {
    hotColdDelta = +(last5 - handicapIndex).toFixed(2);
    if (hotColdDelta < -1.5) form = 'hot';
    else if (hotColdDelta > 1.5) form = 'cold';
  }

  return {
    score_count: diffs.length,
    last_10_avg_differential: last10 != null ? +last10.toFixed(2) : null,
    last_5_avg_differential: last5 != null ? +last5.toFixed(2) : null,
    last_round_differential: lastRound != null ? +lastRound.toFixed(2) : null,
    rounds_last_30_days: recentCount,
    hot_or_cold_delta: hotColdDelta,
    form,
  };
}

function normalizeScores(scores) {
  if (!Array.isArray(scores)) return [];
  return scores.slice(0, SCORES_LIMIT).map(s => ({
    played_at: s.played_at || s.posted_at || null,
    course_name: s.course_name || s.facility_name || null,
    tee_name: s.tee_name || null,
    adjusted_gross: s.adjusted_gross_score != null ? Number(s.adjusted_gross_score) : null,
    differential: s.differential != null ? +parseFloat(s.differential).toFixed(1) : null,
    score_type: s.score_type || null,
  }));
}

// ---------- Normalization ----------
function normalizeMember(member, source, currentHi, lowHi, scoresRaw, courses, previousEntry) {
  // Stash low_hi on the member object for computeChanges to use
  member._low_hi_resolved = lowHi;

  let trend = null;
  let delta = null;
  if (previousEntry?.handicap_index != null && currentHi != null) {
    delta = +(currentHi - previousEntry.handicap_index).toFixed(1);
    if (delta < 0) trend = 'down';
    else if (delta > 0) trend = 'up';
    else trend = 'flat';
  }

  const scoreAnalysis = scoresRaw
    ? analyzeScores(scoresRaw, currentHi)
    : { score_count: 0, form: source === 'manual' ? 'unmeasured' : 'unknown',
        last_10_avg_differential: null, last_5_avg_differential: null,
        last_round_differential: null, rounds_last_30_days: 0, hot_or_cold_delta: null };
  const recentScores = scoresRaw ? normalizeScores(scoresRaw) : [];

  return {
    name: member.name,
    team: member.team,
    source, // 'ghin_live', 'ghin_seeded', or 'manual'
    ghin: member.ghin === GHIN_PLACEHOLDER ? null : (member.ghin ?? null),
    handicap_index: currentHi,
    low_hi: lowHi,
    draft_hi: member.draft_hi ?? null,
    fbc12_index: member.fbc12_index ?? null,
    played_fbc12: member.played_fbc12 ?? false,
    changes: computeChanges(currentHi, member),
    course_handicaps: computeCourseHandicaps(currentHi, courses),
    trend,
    delta_since_last_fetch: delta,
    form: scoreAnalysis,
    recent_scores: recentScores,
    manual_note: member.manual_handicap?.note ?? null,
  };
}

// ---------- Team Totals ----------
function computeTeamTotals(golfers, courses) {
  const teams = { forster: [], mangold: [] };
  for (const g of golfers) {
    if (teams[g.team]) teams[g.team].push(g);
  }

  const buildTotal = (members) => {
    const sum = (fn) => members.reduce((acc, m) => {
      const v = fn(m);
      return v != null ? acc + v : acc;
    }, 0);

    const total = {
      hcp: +sum(m => m.handicap_index).toFixed(1),
      low_52wk: +sum(m => m.low_hi).toFixed(1),
      draft: +sum(m => m.draft_hi).toFixed(1),
      fbc12_index: +members
        .filter(m => m.played_fbc12 && m.fbc12_index != null)
        .reduce((acc, m) => acc + m.fbc12_index, 0).toFixed(1),
      changes: {
        vs_52wk_low: { change: null, pct_change: null },
        vs_draft: { change: null, pct_change: null },
        vs_fbc12: { change: null, pct_change: null },
      },
      course_handicaps: {},
    };

    // Total changes (sum of individual changes)
    total.changes.vs_52wk_low.change = +sum(m => m.changes.vs_52wk_low.change).toFixed(1);
    total.changes.vs_52wk_low.pct_change = total.low_52wk !== 0
      ? +(total.changes.vs_52wk_low.change / total.low_52wk).toFixed(4) : null;
    total.changes.vs_draft.change = +sum(m => m.changes.vs_draft.change).toFixed(1);
    total.changes.vs_fbc12.change = +sum(m => m.changes.vs_fbc12.change).toFixed(1);

    // Course handicap totals
    for (const key of Object.keys(courses)) {
      total.course_handicaps[key] = members.reduce((acc, m) => {
        const ch = m.course_handicaps?.[key]?.course_handicap;
        return ch != null ? acc + ch : acc;
      }, 0);
    }

    return total;
  };

  return {
    forster: buildTotal(teams.forster),
    mangold: buildTotal(teams.mangold),
  };
}

// ---------- Main ----------
async function main() {
  // Load roster
  const rosterDoc = await readJson(MEMBERS_FILE);
  const { liveGhinMembers, seededMembers, manualMembers, errors } = classifyMembers(rosterDoc.members || []);

  if (errors.length > 0) {
    log('ERROR', 'Roster validation errors:');
    for (const e of errors) log('ERROR', `  - ${e}`);
    process.exit(1);
  }

  const totalMembers = liveGhinMembers.length + seededMembers.length + manualMembers.length;
  log('INFO', `Loaded ${totalMembers} members: ${liveGhinMembers.length} live GHIN, ${seededMembers.length} seeded, ${manualMembers.length} manual`);

  // Load courses
  const coursesDoc = await readJson(COURSES_FILE);
  const courses = coursesDoc.courses || {};
  log('INFO', `Loaded ${Object.keys(courses).length} courses: ${Object.keys(courses).join(', ')}`);

  // Load previous output for trend
  const previous = await readJsonIfExists(PREVIOUS_FILE);
  const previousByKey = {};
  if (previous?.golfers) {
    for (const g of previous.golfers) {
      previousByKey[g.name] = g;
    }
  }

  const results = [];
  const failures = [];

  // ----- Process manual members (no network) -----
  for (const member of manualMembers) {
    const mh = member.manual_handicap;
    const normalized = normalizeMember(
      member, 'manual',
      mh.handicap_index,
      mh.low_hi ?? mh.handicap_index,
      null, courses,
      previousByKey[member.name]
    );
    results.push(normalized);
    log('INFO', `Manual: ${member.name} → HI ${normalized.handicap_index}`);
  }

  // ----- Process seeded members (no network) -----
  for (const member of seededMembers) {
    const sh = member.seeded_handicap;
    const normalized = normalizeMember(
      member, 'ghin_seeded',
      sh.handicap_index,
      sh.low_hi ?? sh.handicap_index,
      null, courses,
      previousByKey[member.name]
    );
    results.push(normalized);
    log('INFO', `Seeded: ${member.name} → HI ${normalized.handicap_index}`);
  }

  // ----- Process live GHIN members (network) -----
  if (liveGhinMembers.length > 0) {
    const email = process.env.GHIN_EMAIL;
    const password = process.env.GHIN_PASSWORD;

    if (!email || !password) {
      log('ERROR', 'GHIN_EMAIL and GHIN_PASSWORD required (have live GHIN members configured)');
      process.exit(1);
    }

    let token;
    try {
      token = await ghinLogin(email, password);
    } catch (err) {
      log('ERROR', 'Authentication failed', err.message);
      process.exit(2);
    }

    for (const member of liveGhinMembers) {
      try {
        const golferRaw = await fetchGolfer(token, member.ghin);
        if (!golferRaw) {
          failures.push({ name: member.name, ghin: member.ghin, reason: 'not_found' });
          // Fall back to seeded data if available
          if (member.seeded_handicap?.handicap_index != null) {
            const sh = member.seeded_handicap;
            const normalized = normalizeMember(
              member, 'ghin_seeded',
              sh.handicap_index, sh.low_hi ?? sh.handicap_index,
              null, courses, previousByKey[member.name]
            );
            results.push({ ...normalized, fallback_reason: 'ghin_not_found' });
            log('WARN', `Using seeded fallback for ${member.name}`);
          }
          continue;
        }

        const internalId = golferRaw.id ?? member.ghin;
        let scoresRaw = [];
        try {
          scoresRaw = await fetchScores(token, internalId, SCORES_LIMIT);
          await new Promise(r => setTimeout(r, 150));
        } catch (err) {
          log('WARN', `Scores fetch errored for ${member.name}: ${err.message}`);
        }

        const hiRaw = golferRaw.handicap_index;
        const hiNum = (hiRaw && hiRaw !== 'NH' && !isNaN(parseFloat(hiRaw)))
          ? parseFloat(hiRaw) : null;
        const lowHiRaw = golferRaw.low_hi;
        const lowHiNum = (lowHiRaw && !isNaN(parseFloat(lowHiRaw)))
          ? parseFloat(lowHiRaw) : null;

        const normalized = normalizeMember(
          member, 'ghin_live',
          hiNum, lowHiNum,
          scoresRaw, courses,
          previousByKey[member.name]
        );
        results.push(normalized);
        log('INFO', `GHIN: ${member.name}: HI ${hiNum}, ${scoresRaw.length} scores, form: ${normalized.form.form}`);

        await new Promise(r => setTimeout(r, 250));
      } catch (err) {
        failures.push({ name: member.name, ghin: member.ghin, reason: err.message });
        log('WARN', `Error fetching ${member.name}: ${err.message}`);
      }
    }

    const failureRate = failures.length / liveGhinMembers.length;
    if (failureRate > 0.5) {
      log('ERROR', `GHIN failure rate ${(failureRate * 100).toFixed(0)}% exceeds threshold — aborting write`);
      process.exit(3);
    }
  }

  // Compute team totals
  const team_totals = computeTeamTotals(results, courses);

  const output = {
    generated_at: new Date().toISOString(),
    source: 'Excel seed + GHIN live',
    courses,
    member_count: results.length,
    live_ghin_count: liveGhinMembers.length - failures.length,
    seeded_count: seededMembers.length + failures.length,
    manual_count: manualMembers.length,
    failure_count: failures.length,
    failures,
    team_totals,
    golfers: results,
  };

  await writeJson(OUTPUT_FILE, output);
  log('INFO', `Wrote ${OUTPUT_FILE}`);
  log('INFO', `Summary: ${results.length} total members`);
}

main().catch(err => {
  log('ERROR', 'Unhandled error', err.stack || err.message);
  process.exit(99);
});
