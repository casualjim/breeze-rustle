# Breeze Search Example

## First, index a codebase:
```bash
breeze index /path/to/your/codebase
```

## Then search for code:
```bash
# Basic search
breeze search "implement authentication"

# Search with more results
breeze search "async function" --limit 20

# Show full file content in results
breeze search "database connection" --full

# Use a different database
breeze search "error handling" --database /path/to/other/db
```

## Example Output:
```
Found 3 results for query: "implement authentication"

1. src/auth/login.rs (score: 0.892)
   Size: 2456 bytes
   Preview:
   use bcrypt::{hash, verify};
   use jsonwebtoken::{encode, decode, Header, Validation};

   pub async fn authenticate_user(username: &str, password: &str) -> Result<Token, AuthError> {
       let user = fetch_user_by_username(username).await?;
   ... (45 more lines)

2. src/middleware/auth.rs (score: 0.847)
   Size: 1823 bytes
   Preview:
   use actix_web::{FromRequest, HttpRequest};
   use futures::future::{Ready, ready};

   pub struct AuthenticatedUser {
       pub id: Uuid,
   ... (32 more lines)

3. tests/auth_test.rs (score: 0.756)
   Size: 3102 bytes
   Preview:
   #[tokio::test]
   async fn test_user_authentication() {
       let app = create_test_app().await;

       let response = app.post("/auth/login")
   ... (58 more lines)
```