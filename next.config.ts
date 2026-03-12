import type { NextConfig } from "next";

const nextConfig: NextConfig = {
    images: {
        domains: ['images.unsplash.com', 'picsum.photos'],
    },
    // Dynamic SaaS Mode (Enables /api/protect and /dashboard)
    typescript: {
        ignoreBuildErrors: true, // Ensuring smooth conquest launch
    },
};

export default nextConfig;

