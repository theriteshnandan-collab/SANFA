"use client";

import { motion } from "framer-motion";

interface LineIllustrationProps {
  className?: string;
}

export default function LineIllustration({ className }: LineIllustrationProps) {
  return (
    <div className={className}>
      <motion.svg
        viewBox="0 0 400 300"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-auto"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: "-100px" }}
      >
        {/* Simple Abstract Line-Art Representing Connectivity/Security */}
        <motion.path
          d="M50 150 C 50 50, 150 50, 150 150 S 250 250, 250 150 S 350 50, 350 150"
          stroke="#FF0066"
          strokeWidth="3"
          strokeLinecap="round"
          variants={{
            hidden: { pathLength: 0, opacity: 0 },
            visible: {
              pathLength: 1,
              opacity: 1,
              transition: { duration: 1.5, ease: "easeInOut" }
            }
          }}
        />
        <motion.rect
          x="140"
          y="110"
          width="20"
          height="80"
          rx="10"
          stroke="#111111"
          strokeWidth="2"
          variants={{
            hidden: { pathLength: 0, opacity: 0 },
            visible: {
              pathLength: 1,
              opacity: 1,
              transition: { duration: 1, delay: 0.5, ease: "easeInOut" }
            }
          }}
        />
        <motion.circle
          cx="250"
          cy="150"
          r="40"
          stroke="#111111"
          strokeWidth="2"
          strokeDasharray="10 5"
          variants={{
            hidden: { pathLength: 0, opacity: 0 },
            visible: {
              pathLength: 1,
              opacity: 1,
              transition: { duration: 1.2, delay: 0.8, ease: "easeInOut" }
            }
          }}
        />
      </motion.svg>
    </div>
  );
}
