// Main entry point - initializes all modules

import { defaultCatalogs } from './constants.js';
import { showToast, initializeFooter, initializeKofi } from './modules/ui.js';
import { initializeNavigation, switchSection, lockNavigationForLoggedOut, initializeMobileNav, updateMobileLayout, unlockNavigation } from './modules/navigation.js';
import { initializeAuth, setStremioLoggedOutState } from './modules/auth.js';
import { initializeTrakt, setTraktLoggedOutState } from './modules/trakt.js';
import { initializeSimklProvider, setSimklLoggedOutState } from './modules/simkl.js';
import { initializeCatalogList, renderCatalogList, getCatalogs, setCatalogs } from './modules/catalog.js';
import { initializeForm, clearErrors } from './modules/form.js';

let catalogsState = JSON.parse(JSON.stringify(defaultCatalogs));

const configForm = document.getElementById('configForm');
const catalogList = document.getElementById('catalogList');
const movieGenreList = document.getElementById('movieGenreList');
const seriesGenreList = document.getElementById('seriesGenreList');
const submitBtn = document.getElementById('submitBtn');
const stremioLoginBtn = document.getElementById('stremioLoginBtn');
const stremioLoginText = document.getElementById('stremioLoginText');
const emailInput = document.getElementById('emailInput');
const passwordInput = document.getElementById('passwordInput');
const emailPwdContinueBtn = document.getElementById('emailPwdContinueBtn');
const languageSelect = document.getElementById('languageSelect');
const configNextBtn = document.getElementById('configNextBtn');
const catalogsNextBtn = document.getElementById('catalogsNextBtn');
const successResetBtn = document.getElementById('successResetBtn');
const btnGetStarted = document.getElementById('btn-get-started');

const navItems = {
    welcome: document.getElementById('nav-welcome'),
    login: document.getElementById('nav-login'),
    config: document.getElementById('nav-config'),
    catalogs: document.getElementById('nav-catalogs'),
    install: document.getElementById('nav-install')
};

const sections = {
    welcome: document.getElementById('sect-welcome'),
    login: document.getElementById('sect-login'),
    config: document.getElementById('sect-config'),
    catalogs: document.getElementById('sect-catalogs'),
    install: document.getElementById('sect-install'),
    success: document.getElementById('sect-success')
};

const mainEl = document.querySelector('main');

function resetApp() {
    if (configForm) configForm.reset();
    clearErrors();
    switchSection('welcome');

    Object.keys(navItems).forEach(key => {
        if (key !== 'login' && key !== 'welcome' && navItems[key]) navItems[key].classList.add('disabled');
    });

    setStremioLoggedOutState();
    setTraktLoggedOutState();
    setSimklLoggedOutState();

    catalogsState = JSON.parse(JSON.stringify(defaultCatalogs));
    setCatalogs(catalogsState);
    renderCatalogList();

    if (configForm) configForm.classList.remove('hidden');
    if (sections.success) sections.success.classList.add('hidden');
}

function initializeWelcomeFlow() {
    if (!btnGetStarted) return;
    let touched = false;
    const handleGetStarted = (e) => {
        if (e.type === 'click' && touched) return;
        if (e.type === 'touchstart') touched = true;
        if (navItems.login) navItems.login.classList.remove('disabled');
        switchSection('login');
    };
    btnGetStarted.addEventListener('click', handleGetStarted);
    btnGetStarted.addEventListener('touchstart', handleGetStarted, { passive: true });
}

document.addEventListener('DOMContentLoaded', () => {
    switchSection('welcome');
    initializeWelcomeFlow();

    initializeNavigation({ navItems, sections, mainEl });
    lockNavigationForLoggedOut();

    setCatalogs(catalogsState);
    initializeCatalogList({ catalogList }, { catalogs: catalogsState, renderCatalogList });

    initializeAuth(
        { stremioLoginBtn, stremioLoginText, emailInput, passwordInput, emailPwdContinueBtn, languageSelect },
        { getCatalogs, renderCatalogList, resetApp }
    );

    initializeTrakt(
        { languageSelect },
        { getCatalogs, renderCatalogList, resetApp }
    );

    // Simkl injects its third provider tab at runtime and owns Simkl OAuth submissions.
    // Initialize before the generic form handler so its capture listener can route
    // authenticated Simkl accounts to /tokens/simkl without disturbing Stremio/Trakt.
    initializeSimklProvider(
        { languageSelect },
        { getCatalogs, renderCatalogList, resetApp }
    );

    initializeForm(
        { configForm, submitBtn, emailInput, passwordInput, languageSelect, movieGenreList, seriesGenreList },
        { getCatalogs, resetApp }
    );

    initializeMobileNav();
    initializeFooter();
    initializeKofi();

    updateMobileLayout();
    window.addEventListener('resize', updateMobileLayout);
    window.addEventListener('orientationchange', updateMobileLayout);

    if (configNextBtn) configNextBtn.addEventListener('click', () => switchSection('catalogs'));
    if (catalogsNextBtn) catalogsNextBtn.addEventListener('click', () => switchSection('install'));

    const resetBtn = document.getElementById('resetBtn');
    if (resetBtn) resetBtn.addEventListener('click', resetApp);
    if (successResetBtn) successResetBtn.addEventListener('click', resetApp);
});

window.resetApp = resetApp;
window.switchSection = switchSection;
window.unlockNavigation = unlockNavigation;
