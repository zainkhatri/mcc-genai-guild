import os
import subprocess
import json

# GitHub repository and branch to push the GitHub Pages site to
GITHUB_REPO = "zainkhatri/mcc-genai-guild"
BRANCH = "gh-pages"

def read_agent_data(agent_file):
    """Reads JSON data from the specified agent file."""
    try:
        with open(agent_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {agent_file} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {agent_file} is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def setup_github_pages():
    """Initializes and pushes the GitHub Pages site."""
    try:
        os.makedirs("github-pages", exist_ok=True)
        os.chdir("github-pages")

        #Initialize Git repo if not already
        if not os.path.exists(".git"):
          subprocess.run(["git", "init"], check=True)
          subprocess.run(["git", "remote", "add", "origin", f"https://github.com/{GITHUB_REPO}.git"], check=True)
        
        # Create a simple base HTML file
        with open("index.html", "w") as f:
            f.write(f"""
            <html>
            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <meta name="description" content="Islamic LLM Evaluation Results">
                <meta name="keywords" content="AI, LLM, Islamic, Evaluation">
                <title>Islamic LLM Evaluation</title>
            </head>
            <body>
                <h1>Islamic LLM Evaluation Results</h1>
                <h2>Agent 4 Leaderboard</h2>
                <table border='1'><tr><th>Rank</th><th>Model</th><th>Accuracy</th><th>Ethical Alignment</th><th>Total Score</th><th>Ethical Compliance</th></tr></table>
                <p>Last updated: Never</p>
            </body>
            </html>
            """)
        
        # Commit and push to GitHub Pages
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit for GitHub Pages setup"], check=True)
        subprocess.run(["git", "branch", "-M", BRANCH], check=True)
        subprocess.run(["git", "push", "-u", "origin", BRANCH], check=True, env={"GIT_ASKPASS": "echo", "GIT_USERNAME": os.getenv("GIT_USERNAME"), "GIT_PASSWORD": os.getenv("GIT_PASSWORD")})
    except subprocess.CalledProcessError as e:
        print(f"Error: A subprocess command failed with error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    setup_github_pages()
