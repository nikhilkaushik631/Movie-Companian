/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone', // Enable standalone output for Docker
  images: { 
    domains: ['image.tmdb.org', 'via.placeholder.com'],
    unoptimized: process.env.NODE_ENV === 'development'
  },
  env: {
    NEXT_PUBLIC_API_BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000',
    NEXT_PUBLIC_DEMO: process.env.NEXT_PUBLIC_DEMO || '0'
  },
  // Optimize for production
  experimental: {
    optimizePackageImports: ['@headlessui/react', '@heroicons/react']
  }
};

module.exports = nextConfig;

