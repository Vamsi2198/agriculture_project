Agentic AI Agriculture + Finance Top-up (n8n-based)

Empowering 140M+ Indian smallholder farmers with AI-driven agricultural knowledge and instant credit scoring.

Problem Statement

Farmers in India face several systemic barriers that impact their productivity and financial wellbeing:

Language barriers: 20+ languages with high colloquial usage

Low financial inclusion: only 36% of farmers have access to formal credit (2024 data)

Fragmented agricultural data: scattered across PDFs, portals, and government circulars

Real-time needs: crop health updates, pest outbreak alerts, soil input recommendations, and weather shifts

Low digital literacy: preference for WhatsApp over apps or web browsers
Our Solution

An agentic, multilingual, fact-grounded agriculture advisor combined with a credit scoring engine called TerraCred™. This system is deployed on low-cost infrastructure (n8n + Supabase + free APIs) and is accessible directly via WhatsApp, removing the need for standalone apps.

Features for Farmers

Crop and pest guidance delivered in their native language

Weather and rainfall forecasts

Soil and fertilizer recommendations

Instant credit limit and loan approval scoring via TerraCred™

Access via WhatsApp, enabling ease of use for low digital literacy users
How It Works
1. Agentic Agriculture Advisor

Multilingual Pipeline: Uses Sarvam API to auto-detect and translate queries from multiple Indian languages

Main Agriculture Agent: Routes queries to specialized domain agents (crop, pest, soil, weather, policy, finance, market)

Fact-Grounded Retrieval: Utilizes Supabase vector database to provide accurate, hallucination-free responses

Follow-up Engagement: GPT-3.5 formats responses and proactively asks smart follow-up questions to farmers

Knowledge Ingestion System: Allows uploading of PDFs/CSVs to Google Drive, which are then automatically embedded and stored in Supabase for knowledge retrieval

Farmers can ask questions in Hinglish, Tamil, or other local languages and receive grounded, accurate, and conversational advice.

Technology Stack

Automation: n8n workflow automation platform

Database: Supabase vector DB for embedding and retrieval

Language Detection & Translation: Sarvam API

Language Model: GPT-3.5 for conversational AI and response generation

Communication Channel: WhatsApp for user interaction

Knowledge Storage: Google Drive integration for data ingestion

Impact

This solution addresses key challenges faced by Indian smallholder farmers by combining multilingual AI advisory with instant credit scoring, thereby improving agricultural outcomes and financial inclusion.
