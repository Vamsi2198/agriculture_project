# 🌾 Agriculture Agent Workflow Project for n8n

This project provides a complete, modular system of **agricultural automation agents** built in **n8n**, including:

- One **Main Agriculture Agent** for orchestrating conversations.
- Eight specialized **Sub-Agents** for handling various agricultural topics.
- Supporting files for configuration and messaging interface (WhatsApp).

---

##  Repository Files

| File Name                         | Purpose |
|----------------------------------|---------|
| `[main Agriculture Agent.json](https://github.com/Vamsi2198/agriculture_project/blob/main/main%20Agriculture%20Agent.json)`    | Main agent that handles incoming messages and routes them |
| `crop and varieties agent.json`  | Advises on crop types and varieties |
| `finance and policy agent.json`  | Responds to policy, scheme, and subsidy queries |
| `market and post harvest agent.json` | Provides market prices and post-harvest support |
| `pest and disease agent.json`    | Diagnoses pests/diseases and gives solutions |
| `soil agent.json`                | Offers soil-related advice |
| `weather agent2.json`            | Provides weather updates |
| `weather And rainfall Agent.json`| Gives rainfall predictions |
| `credit limit agent.json`        | Responds to credit and loan-related queries |
| `credentials`                    | Stores credentials for **Vector DB** and **n8n environment** |
| `agriculture workflow.png`       | Visual overview of the workflow architecture |
| `whatsApp QR code.png`           | QR code to connect your WhatsApp number with n8n (if using WhatsApp automation) |

---

##  Requirements

- [n8n](https://n8n.io) installed (locally or via cloud)
- Active accounts & credentials for:
  - OpenAI (or any LLM provider)
  - Weather API
  - Vector Database (e.g., Pinecone, Weaviate, etc.)
  - WhatsApp Business API (or WhatsApp Gateway)

---

##  How to Import Workflows into n8n

1. Open your n8n editor.
2. Click on the menu (☰) → `Import Workflow`.
3. Upload each `.json` file one at a time.
4. Save and enable each workflow.
5. Start with the sub-agents first, then enable the `main Agriculture Agent`.

---

## 🔧 Setup Instructions

Once you have imported the workflows, you need to configure the credentials properly:

- The **Vector Database credentials** and **OpenAI API key** are already mentioned in the [`credentials` file](https://github.com/Vamsi2198/agriculture_project/blob/main/credentials).  
- Make sure to update and save these credentials in your n8n instance so the workflows can access the Vector DB and OpenAI services correctly.

If you find importing and setting up the workflows difficult, you can also use the pre-configured workflows directly from my n8n cloud account:

- Access the workflows here:  
  [https://vamsimulaworkmail2.app.n8n.cloud/workflow/qpLxuPprrKlzrWF3/9ed195](https://vamsimulaworkmail2.app.n8n.cloud/workflow/qpLxuPprrKlzrWF3/9ed195)

- Credentials for Vector DB and OpenAI are already set up in that environment, as mentioned in the [`credentials` file](https://github.com/Vamsi2198/agriculture_project/blob/main/credentials).

This way, you can start testing immediately without the hassle of importing or configuring.

---

##  Agent Workflow Process

The agriculture agent system accepts inputs from two sources: **Web URL** or **WhatsApp messages**. Regardless of the source, the message is processed through the same unified workflow:

1. **Input Handling**:
   - A message is received either through a **Webhook (Web interface)** or from **WhatsApp (via Twilio)**.
   - The input is treated uniformly, no matter which channel it comes from.

2. **Language Detection and Translation**:
   - The input language is detected automatically.
   - If the input is **not in English**, it is translated into English.
   - This is necessary because the **Vector DB stores all agricultural data in English**, which ensures accurate matching and retrieval.

3. **Main Orchestration**:
   - The translated input is passed to the **Main Agriculture Orchestration Agent**.
   - This agent extracts the **intent** of the message (e.g., soil query, weather info, pest issue).
   - Based on the intent, it **calls the appropriate sub-agent** (one of the 8 available).

4. **Sub-Agent Processing**:
   - Each sub-agent specializes in a particular domain (e.g., crop variety, weather, soil, policy).
   - The orchestration agent interacts with sub-agents as needed to **collect relevant data**.
   - This interaction continues **until a meaningful response** is obtained.

5. **Follow-Up Generation**:
   - After receiving the main response, a **follow-up question generation agent** analyzes the output.
   - It generates **suggested follow-up questions** for the farmer, based on the context and the returned data.
   - These questions help guide the farmer for further clarification or next steps.

6. **Response Finalization**:
   - The **main response and follow-up suggestions** are merged into a single output.
   - The original user language (detected earlier) is used to **translate the final response** back from English (if needed), using **Sarvam Translate (free API)**.

7. **Message Delivery**:
   - If the message came from the **web interface**, the final response is returned directly to the chat.
   - If the message came from **WhatsApp**, the response is sent using the **Twilio API** (which supports free WhatsApp messaging in trial accounts).

---

##  Workflow Overview

- Visual reference of the whole system:

![Workflow Diagram](agriculture%20workflow.png)

---

##  How to Use This Workflow with WhatsApp or Web

To start using the agriculture agent workflow:

### WhatsApp
1. Scan the **WhatsApp QR code** image attached in this repository:  
   [whatsApp QR code](https://github.com/Vamsi2198/agriculture_project/blob/main/whatsApp%20QR%20code.png)

2. This will link your WhatsApp number to the workflow via Twilio or your WhatsApp Business API setup.

3. Send a message to the linked WhatsApp number and receive agricultural assistance in your language.

### Web Chat
You can also chat through the web interface using this webhook URL:

[https://vamsimulaworkmail2.app.n8n.cloud/webhook/ede74737-0e2e-42ac-9fbc-2a5aeed8d930/chat](https://vamsimulaworkmail2.app.n8n.cloud/webhook/ede74737-0e2e-42ac-9fbc-2a5aeed8d930/chat)

Just open the link and send your queries directly to the agriculture agent system.

---

##  Testing Your Setup

If you have your own n8n instance, you can import the workflows and test them directly.

Alternatively, you can use the pre-configured n8n environment with credentials provided in the [`credentials` file](https://github.com/Vamsi2198/agriculture_project/blob/main/credentials).

To test:

- Send messages to the WhatsApp number linked via the QR code or
- Use the web chat URL:  
  [https://vamsimulaworkmail2.app.n8n.cloud/webhook/ede74737-0e2e-42ac-9fbc-2a5aeed8d930/chat](https://vamsimulaworkmail2.app.n8n.cloud/webhook/ede74737-0e2e-42ac-9fbc-2a5aeed8d930/chat)

Try sending messages in different Indian languages. The system will:

- Detect the language
- Translate the input to English for processing
- Route the query to appropriate sub-agents
- Generate responses and follow-up questions
- Translate the final reply back into the user's original language

Observe how the agents coordinate and handle multi-lingual inputs seamlessly.

---



##  Contributing

Feel free to fork this repo, improve it, and contribute back with PRs. You can also adapt it for your local agricultural ecosystem.
