// Simkl Authentication Module

import { showToast } from './ui.js';
import { switchSection, unlockNavigation } from './navigation.js';

const SIMKL_STORAGE_KEY = 'watchly_simkl_auth';
let languageSelect = null;
let getCatalogs = null;
let renderCatalogList = null;
let resetApp = null;

export function initializeSimklProvider(domElements, catalogState) {
    languageSelect = domElements.languageSelect;
    getCatalogs = catalogState.getCatalogs;
    renderCatalogList = catalogState.renderCatalogList;
    resetApp = catalogState.resetApp;

    injectSimklUi();
    initializeConnectButton();
    initializeLogoutButton();
    initializeSubmitOverride();
    attemptAutoLogin();
}

export function setSimklLoggedOutState() {
    clearSimklStorage();
    hideStatus();
}

export function getSimklTokensFromStorage() {
    return getStoredAuth();
}

function saveAuth(auth) {
    try { localStorage.setItem(SIMKL_STORAGE_KEY, JSON.stringify(auth)); } catch (e) { console.warn(e); }
}

function getStoredAuth() {
    try {
        const value = localStorage.getItem(SIMKL_STORAGE_KEY);
        return value ? JSON.parse(value) : null;
    } catch (e) {
        clearSimklStorage();
        return null;
    }
}

function clearSimklStorage() {
    try { localStorage.removeItem(SIMKL_STORAGE_KEY); } catch (e) { }
}

function selectedProvider() {
    try { return localStorage.getItem('watchly_login_tab') || 'stremio'; } catch (e) { return 'stremio'; }
}

function injectSimklUi() {
    if (document.getElementById('tabSimkl')) return;

    const tabTrakt = document.getElementById('tabTrakt');
    const tabs = tabTrakt?.parentElement;
    const panelTrakt = document.getElementById('panelTrakt');
    if (!tabs || !panelTrakt) return;

    const tab = document.createElement('button');
    tab.type = 'button';
    tab.id = 'tabSimkl';
    tab.dataset.tab = 'simkl';
    tab.className = 'login-tab flex-1 py-2.5 px-4 rounded-lg text-sm font-medium transition-all text-slate-400 hover:text-white';
    tab.innerHTML = '<span class="flex items-center justify-center gap-2"><span class="w-4 h-4 rounded bg-[#1f9cff] text-white text-[10px] font-bold flex items-center justify-center">S</span>Simkl</span>';
    tabs.appendChild(tab);

    const panel = document.createElement('div');
    panel.id = 'panelSimkl';
    panel.className = 'hidden';
    panel.innerHTML = `
        <div id="simklStatusSection" class="hidden mb-4">
            <div class="flex items-center justify-between gap-4 p-4 bg-neutral-800/60 rounded-xl border border-white/10">
                <div class="flex items-center gap-3 flex-grow min-w-0">
                    <div id="simklStatusAvatar" class="w-10 h-10 rounded-full bg-[#1f9cff] text-white flex items-center justify-center font-bold">S</div>
                    <div class="min-w-0"><div class="text-xs text-slate-500">Connected as</div><div id="simklStatusDisplay" class="text-sm text-white font-medium truncate"></div></div>
                </div>
                <button type="button" id="simklLogoutBtn" class="bg-red-600 hover:bg-red-700 text-white py-2 px-3 rounded-xl text-sm">Disconnect</button>
            </div>
        </div>
        <button type="button" id="simklConnectBtn" class="w-full bg-[#1f9cff] hover:bg-[#1684d9] text-white font-medium py-4 rounded-xl transition flex items-center justify-center gap-3 border border-[#1475bf] shadow-lg">
            <span class="w-6 h-6 rounded bg-white text-[#1f9cff] font-bold flex items-center justify-center">S</span>
            <span class="btn-text text-lg">Connect with Simkl</span>
            <div class="loader hidden w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
        </button>
        <div class="mt-4 text-xs leading-relaxed bg-blue-500/10 border border-blue-500/30 text-blue-200 rounded-xl p-3">
            <strong class="text-blue-300">How it works</strong>
            <span class="block mt-1">Watchly reads your Simkl watch history and personal ratings. Ratings of 10 are strongly weighted, 7–9 are lightly weighted, 3–6 are watched-only, and 1–2 are excluded from recommendation anchors.</span>
        </div>`;
    panelTrakt.insertAdjacentElement('afterend', panel);

    tab.addEventListener('click', () => switchTab('simkl'));
    document.getElementById('tabStremio')?.addEventListener('click', () => switchTab('stremio'));
    document.getElementById('tabTrakt')?.addEventListener('click', () => switchTab('trakt'));

    fetch('/tokens/simkl/config').then(r => r.json()).then(data => {
        if (!data.configured) tab.style.display = 'none';
    }).catch(() => {});
}

function switchTab(tabName) {
    document.querySelectorAll('.login-tab').forEach(tab => {
        const active = tab.dataset.tab === tabName;
        tab.classList.toggle('bg-white', active);
        tab.classList.toggle('text-black', active);
        tab.classList.toggle('shadow', active);
        tab.classList.toggle('text-slate-400', !active);
        tab.classList.toggle('hover:text-white', !active);
    });
    ['stremio', 'trakt', 'simkl'].forEach(name => {
        document.getElementById(`panel${name[0].toUpperCase()}${name.slice(1)}`)?.classList.toggle('hidden', name !== tabName);
    });
    try { localStorage.setItem('watchly_login_tab', tabName); } catch (e) { }
}

function initializeConnectButton() {
    const btn = document.getElementById('simklConnectBtn');
    if (!btn) return;
    btn.addEventListener('click', async () => {
        setConnecting(true);
        try {
            const response = await fetch('/tokens/simkl/authorize');
            const body = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(body.detail || 'Failed to start Simkl authorization');
            const auth = await openPopup(body.url);
            const identity = await fetchIdentity(auth);
            try { localStorage.removeItem('watchly_trakt_auth'); } catch (e) { }
            saveAuth(auth);
            switchTab('simkl');
            showStatus(identity.display || identity.username || 'Simkl User');
            if (identity.exists && identity.settings) populateSettings(identity.settings);
            unlockNavigation();
            switchSection('config');
        } catch (error) {
            showToast(error.message || 'Simkl login failed', 'error');
        } finally {
            setConnecting(false);
        }
    });
}

function initializeLogoutButton() {
    document.getElementById('simklLogoutBtn')?.addEventListener('click', () => {
        if (resetApp) resetApp();
    });
}

function openPopup(url) {
    return new Promise((resolve, reject) => {
        const popup = window.open(url, 'simkl_oauth', 'width=600,height=700,resizable=yes,scrollbars=yes');
        if (!popup) return reject(new Error('Please allow popups for this site.'));
        let settled = false;
        const timer = setInterval(() => {
            if (popup.closed && !settled) {
                settled = true; cleanup(); reject(new Error('Authorization window was closed'));
            }
        }, 500);
        function onMessage(event) {
            if (event.origin !== window.location.origin || !event.data) return;
            if (event.data.type === 'simkl_auth_success') {
                settled = true; cleanup(); resolve({ access_token: event.data.access_token });
            } else if (event.data.type === 'simkl_auth_error') {
                settled = true; cleanup(); reject(new Error(event.data.error || 'Simkl authorization failed'));
            }
        }
        function cleanup() { clearInterval(timer); window.removeEventListener('message', onMessage); }
        window.addEventListener('message', onMessage);
    });
}

async function fetchIdentity(auth) {
    const response = await fetch('/tokens/simkl/identity', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ simkl_access_token: auth.access_token })
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || 'Failed to verify Simkl account');
    return body;
}

async function attemptAutoLogin() {
    const auth = getStoredAuth();
    if (!auth?.access_token || selectedProvider() !== 'simkl') return;
    try {
        const identity = await fetchIdentity(auth);
        showStatus(identity.display || identity.username || 'Simkl User');
        if (identity.exists && identity.settings) populateSettings(identity.settings);
        unlockNavigation();
        switchSection('config');
    } catch (error) {
        clearSimklStorage();
    }
}

function initializeSubmitOverride() {
    const submit = document.getElementById('submitBtn');
    if (!submit) return;
    submit.addEventListener('click', async event => {
        const auth = getStoredAuth();
        if (!auth?.access_token || selectedProvider() !== 'simkl') return;
        event.preventDefault();
        event.stopImmediatePropagation();
        await submitSimkl(auth, submit);
    }, true);
}

async function submitSimkl(auth, submit) {
    const text = submit.querySelector('.btn-text');
    const loader = submit.querySelector('.loader');
    submit.disabled = true; text?.classList.add('hidden'); loader?.classList.remove('hidden');
    try {
        const tmdb = document.getElementById('tmdbApiKey')?.value.trim();
        if (!tmdb) throw new Error('TMDB API key is required.');
        const posterProvider = document.getElementById('posterRatingProvider')?.value || '';
        const posterKey = document.getElementById('posterRatingApiKey')?.value.trim() || '';
        const catalogs = (getCatalogs ? getCatalogs() : []).map(c => ({
            id: c.id, name: c.name, enabled: c.enabled !== false,
            enabled_movie: c.enabledMovie !== false, enabled_series: c.enabledSeries !== false,
            display_at_home: c.display_at_home !== false, shuffle: c.shuffle === true
        }));
        const payload = {
            simkl_access_token: auth.access_token,
            catalogs,
            language: languageSelect?.value || 'en-US',
            year_min: Number(document.getElementById('yearMin')?.value || 1980),
            year_max: Number(document.getElementById('yearMax')?.value || 2026),
            popularity: document.getElementById('popularitySelect')?.value || 'balanced',
            sorting_order: document.getElementById('sortingOrderSelect')?.value || 'default',
            poster_rating: posterProvider && posterKey ? { provider: posterProvider, api_key: posterKey } : null,
            tmdb_api_key: tmdb,
            openrouter_api_key: document.getElementById('openrouterApiKey')?.value.trim() || null,
            excluded_movie_genres: Array.from(document.querySelectorAll('input[name="movie-genre"]:checked')).map(x => x.value),
            excluded_series_genres: Array.from(document.querySelectorAll('input[name="series-genre"]:checked')).map(x => x.value)
        };
        const response = await fetch('/tokens/simkl/', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
        });
        const body = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(body.detail || 'Failed to create Simkl manifest');
        showSuccess(body.manifestUrl);
    } catch (error) {
        showToast(error.message || 'Simkl setup failed', 'error');
    } finally {
        submit.disabled = false; text?.classList.remove('hidden'); loader?.classList.add('hidden');
    }
}

function showSuccess(url) {
    ['welcome','login','config','catalogs','install','success'].forEach(name => document.getElementById(`sect-${name}`)?.classList.add('hidden'));
    const success = document.getElementById('sect-success');
    if (success) success.classList.remove('hidden');
    const addonUrl = document.getElementById('addonUrl');
    if (addonUrl) addonUrl.textContent = url;
}

function populateSettings(s) {
    if (s.language && languageSelect) languageSelect.value = s.language;
    const assign = (id, value) => { const el = document.getElementById(id); if (el && value !== undefined && value !== null) el.value = value; };
    assign('popularitySelect', s.popularity); assign('yearMin', s.year_min); assign('yearMax', s.year_max);
    assign('sortingOrderSelect', s.sorting_order); assign('tmdbApiKey', s.tmdb_api_key); assign('openrouterApiKey', s.openrouter_api_key);
    if (window.updateYearSlider) window.updateYearSlider();
    if (s.catalogs && getCatalogs) {
        const local = getCatalogs();
        s.catalogs.forEach(remote => {
            const item = local.find(c => c.id === remote.id);
            if (!item) return;
            item.enabled = remote.enabled;
            item.name = remote.name || item.name;
            item.enabledMovie = remote.enabled_movie;
            item.enabledSeries = remote.enabled_series;
            item.display_at_home = remote.display_at_home;
            item.shuffle = remote.shuffle;
        });
        if (renderCatalogList) renderCatalogList();
    }
}

function showStatus(name) {
    const status = document.getElementById('simklStatusSection');
    const display = document.getElementById('simklStatusDisplay');
    const avatar = document.getElementById('simklStatusAvatar');
    const connect = document.getElementById('simklConnectBtn');

    const rawName = String(name || '').trim();
    const visibleName = !rawName || /^\\d+$/.test(rawName) ? 'Simkl User' : rawName;
    const words = visibleName.split(/\\s+/).filter(Boolean);
    const initials = visibleName === 'Simkl User'
        ? 'S'
        : words.slice(0, 2).map(word => word[0]).join('').toUpperCase();

    if (display) display.textContent = visibleName;
    if (avatar) avatar.textContent = initials || 'S';
    status?.classList.remove('hidden');
    connect?.classList.add('hidden');
    switchTab('simkl');
}

function hideStatus() {
    document.getElementById('simklStatusSection')?.classList.add('hidden');
    document.getElementById('simklConnectBtn')?.classList.remove('hidden');
}

function setConnecting(loading) {
    const btn = document.getElementById('simklConnectBtn');
    if (!btn) return;
    btn.disabled = loading;
    btn.querySelector('.btn-text')?.classList.toggle('hidden', loading);
    btn.querySelector('.loader')?.classList.toggle('hidden', !loading);
}
