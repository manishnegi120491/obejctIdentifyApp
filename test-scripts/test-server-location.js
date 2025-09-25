// Test script to verify server.js location and dependencies
const fs = require('fs');
const path = require('path');

console.log('=== Server Location Test ===');
console.log('Current working directory:', process.cwd());
console.log('__dirname:', __dirname);
console.log('process.argv[1]:', process.argv[1]);

// Check if server.js exists in current directory
const serverPath = path.join(process.cwd(), 'server.js');
console.log('Looking for server.js at:', serverPath);
console.log('server.js exists:', fs.existsSync(serverPath));

// Check if package.json exists
const packagePath = path.join(process.cwd(), 'package.json');
console.log('Looking for package.json at:', packagePath);
console.log('package.json exists:', fs.existsSync(packagePath));

// Check if node_modules exists
const nodeModulesPath = path.join(process.cwd(), 'node_modules');
console.log('Looking for node_modules at:', nodeModulesPath);
console.log('node_modules exists:', fs.existsSync(nodeModulesPath));

// Check if express is installed
const expressPath = path.join(process.cwd(), 'node_modules', 'express');
console.log('Looking for express at:', expressPath);
console.log('express exists:', fs.existsSync(expressPath));

// List files in current directory
console.log('\n=== Files in current directory ===');
try {
  const files = fs.readdirSync(process.cwd());
  files.forEach(file => {
    const stat = fs.statSync(path.join(process.cwd(), file));
    console.log(`${file} (${stat.isDirectory() ? 'directory' : 'file'})`);
  });
} catch (error) {
  console.error('Error reading directory:', error.message);
}

console.log('\n=== Environment Variables ===');
console.log('NODE_ENV:', process.env.NODE_ENV);
console.log('PORT:', process.env.PORT);
console.log('PWD:', process.env.PWD);
