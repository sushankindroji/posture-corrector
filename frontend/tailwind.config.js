/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#6366f1", // Indigo
        secondary: "#4f46e5", // Darker indigo
        accent: "#10b981", // Emerald
        background: "#0f172a", // Dark slate
        "background-light": "#1e293b", // Slightly lighter slate
        "background-card": "#1e293b", // Card background
        "text-primary": "#f8fafc", // Almost white
        "text-secondary": "#cbd5e1", // Light gray
        "text-muted": "#64748b", // Muted text
      },
      boxShadow: {
        'card': '0 4px 6px -1px rgba(0, 0, 0, 0.25), 0 2px 4px -1px rgba(0, 0, 0, 0.1)',
        'card-hover': '0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.15)',
      },
      backgroundImage: {
        'hero-pattern': 'linear-gradient(to right bottom, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.98))',
      },
    },
  },
  plugins: [],
}