#!/usr/bin/env python3
"""
DummyJSON API Playground
------------------------

A terminal-only Python script for playing with the DummyJSON fake REST API.

Run:
    python dummyjson_playground.py

No external packages are required.

This script demonstrates:
- API health test
- Products: list, search, get one, categories
- Users: list, get one, user's carts/posts/todos
- Carts: list, get one
- Posts: list, search, get comments
- Todos: list, get one
- Auth-style login demo
- Fake create/update/delete requests

DummyJSON is a fake REST API for development/testing.
Write operations return simulated responses; they do not persist as a real backend database.
"""

from __future__ import annotations

import getpass
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional


BASE_URL = "https://dummyjson.com"


# -----------------------------------------------------------------------------
# HTTP HELPERS
# -----------------------------------------------------------------------------

def pretty_print(data: Any, title: Optional[str] = None, max_chars: int = 12000) -> None:
    """Pretty-print JSON data, truncating very large responses."""
    if title:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    text = json.dumps(data, indent=2, ensure_ascii=False)

    if len(text) > max_chars:
        print(text[:max_chars])
        print(f"\n... output truncated at {max_chars} characters ...")
    else:
        print(text)


def request_json(
    endpoint: str,
    method: str = "GET",
    data: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
    timeout_s: float = 12.0,
) -> Any:
    """
    Send an HTTP request to DummyJSON and return decoded JSON.

    endpoint examples:
        /products
        /products/search?q=phone
        /users/1/carts
        /auth/login
    """

    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint

    url = BASE_URL + endpoint

    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "Accept": "application/json",
        "User-Agent": "dummyjson-python-playground/1.0",
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")

    request = urllib.request.Request(
        url=url,
        data=body,
        headers=headers,
        method=method.upper(),
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            response_body = response.read().decode("utf-8")

            if response_body:
                return json.loads(response_body)

            return {
                "status": response.status,
                "message": "No response body",
            }

    except urllib.error.HTTPError as error:
        response_text = error.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            parsed = response_text

        return {
            "error": True,
            "status": error.code,
            "reason": error.reason,
            "response": parsed,
        }

    except urllib.error.URLError as error:
        return {
            "error": True,
            "message": str(error.reason),
        }

    except TimeoutError:
        return {
            "error": True,
            "message": "Request timed out.",
        }


def ask_int(prompt: str, default: int, low: Optional[int] = None, high: Optional[int] = None) -> int:
    """Prompt for an integer with bounds and a default."""
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            print("Invalid number. Using default.")
            value = default

    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)

    return value


def ask_text(prompt: str, default: str = "") -> str:
    """Prompt for text with a default."""
    raw = input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    return raw if raw else default


def make_query(params: Dict[str, Any]) -> str:
    """Build a URL query string from non-empty parameters."""
    clean = {k: v for k, v in params.items() if v is not None and v != ""}
    return urllib.parse.urlencode(clean)


# -----------------------------------------------------------------------------
# PRODUCTS
# -----------------------------------------------------------------------------

def list_products() -> None:
    limit = ask_int("Limit", 10, 0, 100)
    skip = ask_int("Skip", 0, 0, 1000)
    select = ask_text("Select fields comma-separated", "title,price,brand,category,rating")

    query = make_query({
        "limit": limit,
        "skip": skip,
        "select": select,
    })

    data = request_json(f"/products?{query}")
    pretty_print(data, "Products")


def search_products() -> None:
    q = ask_text("Search products for", "phone")
    limit = ask_int("Limit", 10, 0, 100)
    query = make_query({"q": q, "limit": limit})
    data = request_json(f"/products/search?{query}")
    pretty_print(data, f"Product search: {q}")


def get_product() -> None:
    product_id = ask_int("Product ID", 1, 1)
    data = request_json(f"/products/{product_id}")
    pretty_print(data, f"Product {product_id}")


def list_product_categories() -> None:
    data = request_json("/products/categories")
    pretty_print(data, "Product categories")


def list_products_by_category() -> None:
    category = ask_text("Category slug", "smartphones")
    limit = ask_int("Limit", 10, 0, 100)
    query = make_query({"limit": limit})
    data = request_json(f"/products/category/{urllib.parse.quote(category)}?{query}")
    pretty_print(data, f"Products in category: {category}")


def create_fake_product() -> None:
    title = ask_text("New product title", "API Playground Product")
    price = ask_int("Price", 99, 1)
    description = ask_text("Description", "Created from a Python DummyJSON playground.")
    category = ask_text("Category", "smartphones")

    payload = {
        "title": title,
        "price": price,
        "description": description,
        "category": category,
    }

    data = request_json("/products/add", method="POST", data=payload)
    pretty_print(data, "POST /products/add simulated response")


def update_fake_product() -> None:
    product_id = ask_int("Product ID to update", 1, 1)
    title = ask_text("Updated title", "Updated API Playground Product")
    price = ask_int("Updated price", 149, 1)

    payload = {
        "title": title,
        "price": price,
    }

    data = request_json(f"/products/{product_id}", method="PUT", data=payload)
    pretty_print(data, f"PUT /products/{product_id} simulated response")


def patch_fake_product() -> None:
    product_id = ask_int("Product ID to patch", 1, 1)
    field = ask_text("Field to patch", "title")
    value = ask_text("New value", "Partially Patched Product")

    payload = {
        field: value,
    }

    data = request_json(f"/products/{product_id}", method="PATCH", data=payload)
    pretty_print(data, f"PATCH /products/{product_id} simulated response")


def delete_fake_product() -> None:
    product_id = ask_int("Product ID to delete", 1, 1)
    data = request_json(f"/products/{product_id}", method="DELETE")
    pretty_print(data, f"DELETE /products/{product_id} simulated response")


# -----------------------------------------------------------------------------
# USERS
# -----------------------------------------------------------------------------

def list_users() -> None:
    limit = ask_int("Limit", 10, 0, 100)
    skip = ask_int("Skip", 0, 0, 1000)
    select = ask_text("Select fields comma-separated", "firstName,lastName,age,email,username")

    query = make_query({
        "limit": limit,
        "skip": skip,
        "select": select,
    })

    data = request_json(f"/users?{query}")
    pretty_print(data, "Users")


def search_users() -> None:
    q = ask_text("Search users for", "John")
    limit = ask_int("Limit", 10, 0, 100)
    query = make_query({"q": q, "limit": limit})
    data = request_json(f"/users/search?{query}")
    pretty_print(data, f"User search: {q}")


def get_user() -> None:
    user_id = ask_int("User ID", 1, 1)
    data = request_json(f"/users/{user_id}")
    pretty_print(data, f"User {user_id}")


def get_user_carts() -> None:
    user_id = ask_int("User ID", 1, 1)
    data = request_json(f"/users/{user_id}/carts")
    pretty_print(data, f"Carts for user {user_id}")


def get_user_posts() -> None:
    user_id = ask_int("User ID", 1, 1)
    data = request_json(f"/users/{user_id}/posts")
    pretty_print(data, f"Posts for user {user_id}")


def get_user_todos() -> None:
    user_id = ask_int("User ID", 1, 1)
    data = request_json(f"/users/{user_id}/todos")
    pretty_print(data, f"Todos for user {user_id}")


# -----------------------------------------------------------------------------
# CARTS
# -----------------------------------------------------------------------------

def list_carts() -> None:
    limit = ask_int("Limit", 10, 0, 100)
    skip = ask_int("Skip", 0, 0, 1000)
    query = make_query({"limit": limit, "skip": skip})
    data = request_json(f"/carts?{query}")
    pretty_print(data, "Carts")


def get_cart() -> None:
    cart_id = ask_int("Cart ID", 1, 1)
    data = request_json(f"/carts/{cart_id}")
    pretty_print(data, f"Cart {cart_id}")


def create_fake_cart() -> None:
    user_id = ask_int("User ID", 1, 1)
    product_id = ask_int("Product ID", 1, 1)
    quantity = ask_int("Quantity", 2, 1)

    payload = {
        "userId": user_id,
        "products": [
            {
                "id": product_id,
                "quantity": quantity,
            }
        ],
    }

    data = request_json("/carts/add", method="POST", data=payload)
    pretty_print(data, "POST /carts/add simulated response")


# -----------------------------------------------------------------------------
# POSTS AND COMMENTS
# -----------------------------------------------------------------------------

def list_posts() -> None:
    limit = ask_int("Limit", 10, 0, 100)
    skip = ask_int("Skip", 0, 0, 1000)
    select = ask_text("Select fields comma-separated", "title,body,tags,reactions,userId")

    query = make_query({
        "limit": limit,
        "skip": skip,
        "select": select,
    })

    data = request_json(f"/posts?{query}")
    pretty_print(data, "Posts")


def search_posts() -> None:
    q = ask_text("Search posts for", "love")
    limit = ask_int("Limit", 10, 0, 100)
    query = make_query({"q": q, "limit": limit})
    data = request_json(f"/posts/search?{query}")
    pretty_print(data, f"Post search: {q}")


def get_post() -> None:
    post_id = ask_int("Post ID", 1, 1)
    data = request_json(f"/posts/{post_id}")
    pretty_print(data, f"Post {post_id}")


def get_post_comments() -> None:
    post_id = ask_int("Post ID", 1, 1)
    data = request_json(f"/posts/{post_id}/comments")
    pretty_print(data, f"Comments for post {post_id}")


def create_fake_post() -> None:
    title = ask_text("Post title", "DummyJSON Python Post")
    body = ask_text("Post body", "This fake post was created from the Python playground.")
    user_id = ask_int("User ID", 1, 1)

    payload = {
        "title": title,
        "body": body,
        "userId": user_id,
    }

    data = request_json("/posts/add", method="POST", data=payload)
    pretty_print(data, "POST /posts/add simulated response")


# -----------------------------------------------------------------------------
# TODOS
# -----------------------------------------------------------------------------

def list_todos() -> None:
    limit = ask_int("Limit", 10, 0, 100)
    skip = ask_int("Skip", 0, 0, 1000)
    query = make_query({"limit": limit, "skip": skip})
    data = request_json(f"/todos?{query}")
    pretty_print(data, "Todos")


def get_todo() -> None:
    todo_id = ask_int("Todo ID", 1, 1)
    data = request_json(f"/todos/{todo_id}")
    pretty_print(data, f"Todo {todo_id}")


def create_fake_todo() -> None:
    todo_text = ask_text("Todo text", "Learn DummyJSON from Python")
    completed_raw = ask_text("Completed? yes/no", "no").lower()
    completed = completed_raw in {"y", "yes", "true", "1"}
    user_id = ask_int("User ID", 1, 1)

    payload = {
        "todo": todo_text,
        "completed": completed,
        "userId": user_id,
    }

    data = request_json("/todos/add", method="POST", data=payload)
    pretty_print(data, "POST /todos/add simulated response")


# -----------------------------------------------------------------------------
# AUTH DEMO
# -----------------------------------------------------------------------------

def login_demo() -> None:
    """
    Demonstrate DummyJSON auth login.

    Known sample credentials from DummyJSON docs often include:
        username: emilys
        password: emilyspass

    If those change, check:
        https://dummyjson.com/docs/auth
    """
    print("\nDummyJSON auth demo")
    print("Try the documented sample user if needed: username=emilys, password=emilyspass")

    username = ask_text("Username", "emilys")
    password = getpass.getpass("Password [hidden, press Enter for emilyspass]: ")
    if not password:
        password = "emilyspass"

    payload = {
        "username": username,
        "password": password,
        "expiresInMins": 30,
    }

    data = request_json("/auth/login", method="POST", data=payload)
    pretty_print(data, "POST /auth/login response")

    access_token = None
    if isinstance(data, dict):
        access_token = data.get("accessToken") or data.get("token")

    if access_token:
        print("\nFetching authenticated user with returned token...")
        me = request_json("/auth/me", method="GET", token=access_token)
        pretty_print(me, "GET /auth/me response")
    else:
        print("\nNo token was returned. Login may have failed or the auth response format changed.")


# -----------------------------------------------------------------------------
# GENERIC REQUEST TOOL
# -----------------------------------------------------------------------------

def custom_request() -> None:
    print("\nCustom DummyJSON request")
    print("Examples:")
    print("  /products?limit=5&select=title,price")
    print("  /products/search?q=phone")
    print("  /users/1")
    print("  /posts/1/comments")
    endpoint = ask_text("Endpoint", "/products?limit=5")
    method = ask_text("Method", "GET").upper()

    data = None
    if method in {"POST", "PUT", "PATCH"}:
        print('Enter JSON body, or press Enter for {"title":"Custom request"}')
        raw = input("JSON body: ").strip()
        if raw:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"Invalid JSON: {exc}")
                return
        else:
            data = {"title": "Custom request"}

    response = request_json(endpoint, method=method, data=data)
    pretty_print(response, f"{method} {endpoint}")


def run_all_basic_demos() -> None:
    """Run a compact set of demos without prompts."""
    pretty_print(request_json("/test"), "API health test")
    pretty_print(request_json("/products?limit=3&select=title,price,category,rating"), "First 3 products")
    pretty_print(request_json("/products/search?q=phone&limit=3"), "Search products for phone")
    pretty_print(request_json("/users?limit=3&select=firstName,lastName,email,username"), "First 3 users")
    pretty_print(request_json("/users/1/carts"), "Carts for user 1")
    pretty_print(request_json("/posts?limit=3&select=title,tags,userId"), "First 3 posts")
    pretty_print(request_json("/todos?limit=3"), "First 3 todos")


# -----------------------------------------------------------------------------
# MENU
# -----------------------------------------------------------------------------

MENU_OPTIONS = {
    "1": ("API health test", lambda: pretty_print(request_json("/test"), "API health test")),
    "2": ("List products", list_products),
    "3": ("Search products", search_products),
    "4": ("Get one product", get_product),
    "5": ("List product categories", list_product_categories),
    "6": ("List products by category", list_products_by_category),
    "7": ("Create fake product", create_fake_product),
    "8": ("Update fake product", update_fake_product),
    "9": ("Patch fake product", patch_fake_product),
    "10": ("Delete fake product", delete_fake_product),
    "11": ("List users", list_users),
    "12": ("Search users", search_users),
    "13": ("Get one user", get_user),
    "14": ("Get user's carts", get_user_carts),
    "15": ("Get user's posts", get_user_posts),
    "16": ("Get user's todos", get_user_todos),
    "17": ("List carts", list_carts),
    "18": ("Get one cart", get_cart),
    "19": ("Create fake cart", create_fake_cart),
    "20": ("List posts", list_posts),
    "21": ("Search posts", search_posts),
    "22": ("Get one post", get_post),
    "23": ("Get comments for a post", get_post_comments),
    "24": ("Create fake post", create_fake_post),
    "25": ("List todos", list_todos),
    "26": ("Get one todo", get_todo),
    "27": ("Create fake todo", create_fake_todo),
    "28": ("Auth login demo", login_demo),
    "29": ("Custom request", custom_request),
    "30": ("Run basic demos", run_all_basic_demos),
}


def print_menu() -> None:
    print("\nDummyJSON API Playground")
    print("-" * 80)

    groups = [
        ("Basics", ["1", "29", "30"]),
        ("Products", ["2", "3", "4", "5", "6", "7", "8", "9", "10"]),
        ("Users", ["11", "12", "13", "14", "15", "16"]),
        ("Carts", ["17", "18", "19"]),
        ("Posts", ["20", "21", "22", "23", "24"]),
        ("Todos", ["25", "26", "27"]),
        ("Auth", ["28"]),
    ]

    for title, keys in groups:
        print(f"\n{title}:")
        for key in keys:
            print(f"  {key:>2}. {MENU_OPTIONS[key][0]}")

    print("\n   0. Exit")


def menu_loop() -> None:
    while True:
        print_menu()
        choice = input("\nChoose an option: ").strip()

        if choice == "0":
            print("Exiting.")
            return

        option = MENU_OPTIONS.get(choice)
        if not option:
            print("Invalid option.")
            continue

        label, function = option
        try:
            function()
        except KeyboardInterrupt:
            print("\nInterrupted.")
            return
        except Exception as exc:
            print(f"\nUnexpected error while running '{label}': {exc}")


def main() -> None:
    print("DummyJSON API Playground")
    print("A Python-only terminal client for exploring DummyJSON resources.")
    print("Base URL:", BASE_URL)
    menu_loop()


if __name__ == "__main__":
    main()
