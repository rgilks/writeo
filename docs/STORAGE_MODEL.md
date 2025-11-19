# Storage Model Documentation

**Last Updated:** January 2025

---

## Overview

Writeo uses an **opt-in server storage model** to maximize user privacy and minimize legal compliance requirements. By default, no data is stored on servers.

---

## Default Behavior (No Server Storage)

**When `storeResults: false` (default):**

- ✅ Results are returned immediately in the PUT response
- ✅ Results are stored automatically in browser localStorage
- ✅ No data is stored on servers (R2/KV)
- ✅ Users have full control over their data
- ✅ Data never leaves the user's device

**Benefits:**

- Maximum privacy
- No legal compliance requirements (no server data collection)
- No data retention policies needed
- No deletion/export APIs needed
- User controls data lifecycle

---

## Opt-in Server Storage

**When `storeResults: true` (opt-in):**

- ✅ Results are returned immediately in the PUT response
- ✅ Results are stored in browser localStorage (as always)
- ✅ Results are also stored on servers (R2/KV) for 90 days
- ✅ Results can be accessed from any device via GET endpoint
- ✅ Data is automatically deleted after 90 days

**Use Cases:**

- Users who want to access results from multiple devices
- Users who want server-side backup
- Users who want to share submission IDs with others

---

## Storage Locations

### Browser Storage (Always Used)

- **localStorage**: Persistent storage across sessions
- **sessionStorage**: Temporary storage during navigation
- **Retention**: Until user clears browser data
- **Privacy**: Data never leaves user's device

### Server Storage (Opt-in Only)

- **R2**: Questions, answers, submissions (only if `storeResults: true`)
- **KV**: Assessment results (only if `storeResults: true`)
- **Retention**: 90 days, then automatic deletion
- **Access**: Via GET `/text/submissions/{id}` endpoint

---

## API Behavior

### PUT `/text/submissions/{id}`

**Default (`storeResults: false`):**

- Processes submission
- Returns results immediately
- Stores results in browser localStorage
- Does NOT store on server

**Opt-in (`storeResults: true`):**

- Processes submission
- Returns results immediately
- Stores results in browser localStorage
- Also stores on server (R2/KV)

### GET `/text/submissions/{id}`

**Behavior:**

- Only works if `storeResults: true` was set during submission
- Returns 404 if submission was not stored on server
- Returns results if found (within 90-day retention)

---

## Privacy & Legal Implications

### Default (No Server Storage)

- ✅ No GDPR/CCPA requirements (no server data collection)
- ✅ No COPPA requirements (no data collection from children)
- ✅ No data deletion API needed
- ✅ No data export API needed
- ✅ No data breach notification needed
- ✅ No age verification needed

### Opt-in (Server Storage)

- ⚠️ GDPR/CCPA requirements apply (server data collection)
- ⚠️ Data deletion API recommended
- ⚠️ Data export API recommended
- ⚠️ Data breach notification procedures needed
- ⚠️ Age verification may be needed (if serving children)

---

## Migration Notes

**For existing users:**

- Existing submissions with server storage remain accessible
- New submissions default to no server storage
- Users can opt in to server storage via checkbox

**For developers:**

- API defaults to `storeResults: false`
- Set `storeResults: true` to enable server storage
- GET endpoint only works for opt-in submissions

---

## Best Practices

1. **Default to no storage**: Use `storeResults: false` by default
2. **Make opt-in clear**: Explain benefits of server storage to users
3. **Respect user choice**: Don't force server storage
4. **Provide alternatives**: Browser storage is sufficient for most use cases
5. **Document retention**: Clearly state 90-day retention for opt-in storage

---

**Last Updated:** January 2025
