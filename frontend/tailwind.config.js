/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        base: "#EEF3F9",
        panel: "#FFFFFF",
        accent: "#2563EB",
        accentSoft: "#DBEAFE",
        warm: "#D97706",
        ink: "#0F172A",
        inkMuted: "#475569",
      },
      boxShadow: {
        glow: "0 12px 32px rgba(37, 99, 235, 0.2)",
        soft: "0 18px 40px rgba(15, 23, 42, 0.14)",
      },
      fontFamily: {
        display: ["'Sora'", "'Noto Sans JP'", "sans-serif"],
        sans: ["'Noto Sans JP'", "'Sora'", "sans-serif"],
      },
      animation: {
        rise: "rise 320ms ease-out",
        pulseSoft: "pulseSoft 1.8s ease-in-out infinite",
      },
      keyframes: {
        rise: {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        pulseSoft: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.55" },
        },
      },
    },
  },
  plugins: [],
};
