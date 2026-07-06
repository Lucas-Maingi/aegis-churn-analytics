import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactCompiler: true,
  // Static export: the dashboard compiles to plain HTML/JS in web/out and is
  // served by FastAPI in production (single-container deploy). trailingSlash
  // makes routes export as directory/index.html so a static file server can
  // resolve them.
  output: "export",
  trailingSlash: true,
};

export default nextConfig;
