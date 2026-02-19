import { Navbar } from './components/navbar.js';
import { Footer } from './components/footer.js';
import { Sidebar } from './components/sidebar.js';
import { initSearch } from './search.js';

document.addEventListener('DOMContentLoaded', () => {
    // Initialize Components
    const navbar = new Navbar();
    navbar.render();

    // Initialize Search after Navbar is rendered
    initSearch();

    const footer = new Footer();
    footer.render();

    // Only render sidebar if the container exists (docs pages)
    if (document.getElementById('sidebar')) {
        const sidebar = new Sidebar();
        sidebar.render();
    }

    // Logo Placeholder Generation if missing
    // In a real scenario, we'd expect the image file to exist.
});
