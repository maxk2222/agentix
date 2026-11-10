#!/bin/bash

# Agentix CyberChatbot Widget - Quick Start Script

echo "🚀 Agentix CyberChatbot Widget - Quick Start"
echo "=========================================="
echo ""

# Check if node is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+"
    exit 1
fi

echo "✅ Node.js $(node -v) found"
echo "✅ npm $(npm -v) found"
echo ""

# Create .env.local if not exists
if [ ! -f ".env.local" ]; then
    echo "📝 Creating .env.local..."
    cp .env.example .env.local
    echo "⚠️  Edit .env.local and add your OpenAI API Key"
    echo ""
fi

# Install dependencies
echo "📦 Installing dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "📋 Available commands:"
echo ""
echo "  npm run dev         - Start development server"
echo "  npm run build       - Build for production"
echo "  npm run preview     - Preview production build"
echo "  npm run type-check  - Check TypeScript types"
echo "  npm run lint        - Run linter"
echo ""
echo "🌐 Open http://localhost:5173 in your browser"
echo ""
echo "📚 Documentation:"
echo "  - README.md        - Full documentation"
echo "  - SECURITY.md      - Security guidelines"
echo "  - INSTALLATION.md  - Installation guide"
echo ""
echo "🎯 Next steps:"
echo "1. Edit .env.local with your OpenAI API Key (sk-...)"
echo "2. Run: npm run dev"
echo "3. Open: http://localhost:5173"
echo ""
