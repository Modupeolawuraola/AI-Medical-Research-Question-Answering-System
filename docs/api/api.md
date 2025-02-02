## Endpoints:
/query
/health

# Request

```bash
{
  "query": "string",
  "model": "string",
  "temperature": float
}
```
# Response
```bash
{
  "answer": "string",
  "sources": []
}
```