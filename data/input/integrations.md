# Integrations

## Available Integrations
NovaDesk integrates with a wide range of tools to fit into your existing workflow.

### Slack
Connect NovaDesk to Slack to receive ticket notifications in your Slack channels. Go to **Settings > Integrations > Slack** and click **Connect**. You can configure which events trigger a Slack notification (e.g. new ticket, urgent ticket, ticket resolved).

### Jira
The Jira integration allows you to create Jira issues directly from a NovaDesk ticket. This is useful for escalating bug reports to your engineering team. To enable, go to **Settings > Integrations > Jira** and enter your Jira workspace URL and API token.

### Zapier
NovaDesk is available on Zapier, allowing you to connect to thousands of apps without writing code. Search for "NovaDesk" in the Zapier app directory to get started.

### Google Analytics
Track how customers interact with your Help Centre by connecting Google Analytics. Go to **Settings > Help Centre > Analytics** and paste your Google Analytics Measurement ID.

## API Access
NovaDesk provides a REST API for custom integrations. You can generate an API key from **Settings > API > Generate Key**. Full API documentation is available at docs.novadesk.io/api.

## Webhook Support
NovaDesk supports outgoing webhooks for real-time event notifications. Configure webhooks from **Settings > Integrations > Webhooks**. Supported events include `ticket.created`, `ticket.resolved`, `ticket.assigned`, and `message.received`.
