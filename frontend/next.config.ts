import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Ships a self-contained server with only the files actually imported,
  // which matters on a 2-core box with 3.8 GB of RAM.
  output: "standalone",
};

export default nextConfig;
