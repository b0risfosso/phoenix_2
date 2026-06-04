"""
JSONPlaceholder API Playground

Run:
    python jsonplaceholder_playground.py

This script demonstrates:
- GET all posts
- GET one post
- GET comments for a post
- GET todos for a user
- POST a new fake post
- PUT a full update
- PATCH a partial update
- DELETE a fake post

JSONPlaceholder is a fake REST API.
Write operations return simulated responses, but they do not permanently change server data.
"""

import json
import urllib.request
import urllib.error


BASE_URL = "https://jsonplaceholder.typicode.com"


def print_json(data, title=None):
    """Pretty-print JSON data."""
    if title:
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)

    print(json.dumps(data, indent=2))


def request_json(endpoint, method="GET", data=None):
    """
    Send an HTTP request and return decoded JSON.

    endpoint example:
        "/posts"
        "/posts/1"
        "/posts/1/comments"
    """

    url = BASE_URL + endpoint

    headers = {
        "Content-Type": "application/json; charset=UTF-8"
    }

    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")

    request = urllib.request.Request(
        url=url,
        data=body,
        headers=headers,
        method=method
    )

    try:
        with urllib.request.urlopen(request) as response:
            response_body = response.read().decode("utf-8")

            if response_body:
                return json.loads(response_body)

            return {
                "status": response.status,
                "message": "No response body"
            }

    except urllib.error.HTTPError as error:
        return {
            "error": True,
            "status": error.code,
            "message": error.reason
        }

    except urllib.error.URLError as error:
        return {
            "error": True,
            "message": str(error.reason)
        }


def get_all_posts():
    posts = request_json("/posts")
    print_json(posts[:5], "First 5 Posts")


def get_one_post(post_id):
    post = request_json(f"/posts/{post_id}")
    print_json(post, f"Post {post_id}")


def get_comments_for_post(post_id):
    comments = request_json(f"/posts/{post_id}/comments")
    print_json(comments[:3], f"First 3 Comments for Post {post_id}")


def get_todos_for_user(user_id):
    todos = request_json(f"/users/{user_id}/todos")
    print_json(todos[:5], f"First 5 Todos for User {user_id}")


def create_fake_post():
    new_post = {
        "title": "My fake API post",
        "body": "This post was created from a Python script.",
        "userId": 1
    }

    response = request_json("/posts", method="POST", data=new_post)
    print_json(response, "POST: Create Fake Post")


def update_fake_post(post_id):
    updated_post = {
        "id": int(post_id),
        "title": "Fully updated title",
        "body": "This is a full replacement using PUT.",
        "userId": 1
    }

    response = request_json(f"/posts/{post_id}", method="PUT", data=updated_post)
    print_json(response, f"PUT: Full Update for Post {post_id}")


def patch_fake_post(post_id):
    partial_update = {
        "title": "Partially updated title"
    }

    response = request_json(f"/posts/{post_id}", method="PATCH", data=partial_update)
    print_json(response, f"PATCH: Partial Update for Post {post_id}")


def delete_fake_post(post_id):
    response = request_json(f"/posts/{post_id}", method="DELETE")
    print_json(response, f"DELETE: Fake Delete for Post {post_id}")


def menu():
    while True:
        print("\nJSONPlaceholder API Playground")
        print("-" * 40)
        print("1. Get first 5 posts")
        print("2. Get one post")
        print("3. Get comments for a post")
        print("4. Get todos for a user")
        print("5. Create a fake post")
        print("6. Fully update a fake post")
        print("7. Partially update a fake post")
        print("8. Delete a fake post")
        print("9. Run all demos")
        print("0. Exit")

        choice = input("\nChoose an option: ").strip()

        if choice == "1":
            get_all_posts()

        elif choice == "2":
            post_id = input("Post ID: ").strip()
            get_one_post(post_id)

        elif choice == "3":
            post_id = input("Post ID: ").strip()
            get_comments_for_post(post_id)

        elif choice == "4":
            user_id = input("User ID: ").strip()
            get_todos_for_user(user_id)

        elif choice == "5":
            create_fake_post()

        elif choice == "6":
            post_id = input("Post ID to update: ").strip()
            update_fake_post(post_id)

        elif choice == "7":
            post_id = input("Post ID to patch: ").strip()
            patch_fake_post(post_id)

        elif choice == "8":
            post_id = input("Post ID to delete: ").strip()
            delete_fake_post(post_id)

        elif choice == "9":
            get_all_posts()
            get_one_post(1)
            get_comments_for_post(1)
            get_todos_for_user(1)
            create_fake_post()
            update_fake_post(1)
            patch_fake_post(1)
            delete_fake_post(1)

        elif choice == "0":
            print("Exiting.")
            break

        else:
            print("Invalid option.")


if __name__ == "__main__":
    menu()
