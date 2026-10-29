/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Plus Jakarta Sans Variable"', "Inter", "system-ui", "sans-serif"],
      },
      colors: {
        ink: { DEFAULT: "#16181d", soft: "#3a3f4b", muted: "#5b6170" },
        accent: { DEFAULT: "#5b7cfa", soft: "#dfe6ff", deep: "#3a56c9" },
        idle: "#c9d2e3",
      },
      borderRadius: { card: "24px" },
      boxShadow: {
        glass: "0 10px 40px -12px rgba(40, 48, 80, 0.18), 0 2px 6px -2px rgba(40, 48, 80, 0.06)",
      },
    },
  },
  plugins: [],
};

