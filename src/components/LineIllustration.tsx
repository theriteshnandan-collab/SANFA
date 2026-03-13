"use client";

import { motion, useMotionValue, useSpring, useTransform } from "framer-motion";
import { useEffect } from "react";

interface LineIllustrationProps {
  className?: string;
}

export default function LineIllustration({ className }: LineIllustrationProps) {
  // Mouse tracking logic
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  // Smooth springs for high-end feel
  const springX = useSpring(mouseX, { stiffness: 100, damping: 30 });
  const springY = useSpring(mouseY, { stiffness: 100, damping: 30 });

  // Parallax offsets for different elements
  const pathX = useTransform(springX, [0, 1000], [-15, 15]);
  const pathY = useTransform(springY, [0, 1000], [-10, 10]);
  
  const rectX = useTransform(springX, [0, 1000], [10, -10]);
  const rectY = useTransform(springY, [0, 1000], [5, -5]);

  const circleX = useTransform(springX, [0, 1000], [-5, 5]);
  const circleY = useTransform(springY, [0, 1000], [15, -15]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      mouseX.set(e.clientX);
      mouseY.set(e.clientY);
    };
    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, [mouseX, mouseY]);

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
          style={{ x: pathX, y: pathY }}
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
          style={{ x: rectX, y: rectY }}
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
          style={{ x: circleX, y: circleY }}
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
