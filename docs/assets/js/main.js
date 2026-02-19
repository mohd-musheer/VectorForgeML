import { Navbar } from './components/navbar.js';
import { Footer } from './components/footer.js';
import { Sidebar } from './components/sidebar.js';
// import { initSearch } from './search.js'; // Deferred

document.addEventListener('DOMContentLoaded', () => {
    // 1. Navbar
    const navbar = new Navbar();
    navbar.render();

    // 2. Sidebar (Only if element exists)
    if (document.getElementById('sidebar')) {
        const sidebar = new Sidebar();
        sidebar.render();
    }

    // 3. Footer
    const footer = new Footer();
    footer.render();

    // 4. Performance: Lazy Load Images
    const lazyImages = document.querySelectorAll('img[loading="lazy"]');
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src || img.src;
                    img.classList.add('loaded');
                    imageObserver.unobserve(img);
                }
            });
        });
        lazyImages.forEach(img => imageObserver.observe(img));
    }

    // 5. Typing Animation for Playground (if present)
    const typeTarget = document.querySelector('.typing-demo');
    if (typeTarget) {
        // Simple typing effect
        const text = typeTarget.getAttribute('data-text') || typeTarget.innerText;
        typeTarget.innerText = '';
        let i = 0;
        function type() {
            if (i < text.length) {
                typeTarget.innerText += text.charAt(i);
                i++;
                setTimeout(type, 50);
            }
        }
        type();
    }
});
