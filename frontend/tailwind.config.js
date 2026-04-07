/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        base: "#F0F8FF",
        panel: "#FFFFFF",
        accent: "#0284C7",
        accentSoft: "#BAE6FD",
        accentDeep: "#0369A1",
        teal: "#0891B2",
        warm: "#D97706",
        ink: "#0B1220",
        inkMuted: "#475569",
        inkSubtle: "#94A3B8",
        hairline: "#E2E8F0",
      },
      backgroundImage: {
        "accent-gradient": "linear-gradient(135deg, #38BDF8 0%, #0284C7 60%, #0369A1 100%)",
        "accent-soft-gradient": "linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%)",
        "user-bubble": "linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%)",
      },
      boxShadow: {
        card: "0 1px 2px rgba(15, 23, 42, 0.04), 0 4px 14px rgba(15, 23, 42, 0.06)",
        cardHover: "0 1px 2px rgba(15, 23, 42, 0.06), 0 10px 28px rgba(15, 23, 42, 0.10)",
        floating: "0 10px 30px rgba(2, 132, 199, 0.22), 0 2px 6px rgba(15, 23, 42, 0.06)",
        glow: "0 14px 40px rgba(2, 132, 199, 0.36)",
        soft: "0 18px 40px rgba(15, 23, 42, 0.12)",
      },
      fontFamily: {
        display: ["'Sora'", "'Noto Sans JP'", "sans-serif"],
        sans: ["'Noto Sans JP'", "'Sora'", "sans-serif"],
      },
      animation: {
        rise: "rise 320ms ease-out",
        fadeIn: "fadeIn 240ms ease-out",
        pulseSoft: "pulseSoft 1.8s ease-in-out infinite",
      },
      keyframes: {
        rise: {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
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
