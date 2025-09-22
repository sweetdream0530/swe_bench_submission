import subprocess
import os
import logging

logger = logging.getLogger(__name__)

class PatchGenerator:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self._ensure_git_repo()
    
    def _ensure_git_repo(self):
        """Ensures the given path is a git repository and initializes if not."""
        current_dir = os.getcwd()
        try:
            os.chdir(self.repo_path)
            result = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.info(f"Initializing git repository at {self.repo_path}")
                subprocess.run(["git", "init"], check=True)
                subprocess.run(["git", "config", "user.email", "swebench@example.com"], check=True)
                subprocess.run(["git", "config", "user.name", "SWE-Bench Agent"], check=True)
                # Make an initial commit if no commits exist
                if not self._has_commits():
                    subprocess.run(["git", "add", "."], check=True)
                    subprocess.run(["git", "commit", "-m", "Initial commit by SWE-Bench Agent"], check=True)
        except Exception as e:
            logger.error(f"Error ensuring git repo at {self.repo_path}: {e}")
            raise
        finally:
            os.chdir(current_dir)
    
    def _has_commits(self) -> bool:
        """Checks if the repository has any commits."""
        result = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], 
                              capture_output=True, text=True)
        return result.returncode == 0
    
    def generate_patch(self, commit_message: str = "SWE-Bench fix") -> str:
        """
        Generates a Git patch string for all staged changes in the repository.
        
        Args:
            commit_message (str): The message to use for the temporary commit.
        
        Returns:
            str: A Git patch string.
        """
        current_dir = os.getcwd()
        try:
            os.chdir(self.repo_path)
            
            # Stage all changes
            subprocess.run(["git", "add", "-A"], check=True)
            
            # Create a temporary commit
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
            
            # Generate the diff against the previous commit
            # Check if there's a previous commit
            try:
                subprocess.run(["git", "rev-parse", "HEAD~1"], check=True, capture_output=True)
                # There's a previous commit, diff against it
                patch_command = ["git", "diff", "HEAD~1", "HEAD"]
                result = subprocess.run(patch_command, capture_output=True, text=True, check=True)
                patch_output = result.stdout
            except subprocess.CalledProcessError:
                # No previous commit, diff against empty tree
                patch_command = ["git", "diff", "--cached"]
                result = subprocess.run(patch_command, capture_output=True, text=True, check=True)
                patch_output = result.stdout
            
            # Revert the temporary commit
            try:
                subprocess.run(["git", "rev-parse", "HEAD~1"], check=True, capture_output=True)
                # There's a previous commit, reset to it
                subprocess.run(["git", "reset", "--soft", "HEAD~1"], check=True)
            except subprocess.CalledProcessError:
                # No previous commit, reset to empty
                subprocess.run(["git", "reset", "--hard", "HEAD"], check=True)
            
            return patch_output
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error generating patch: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during patch generation: {e}")
            raise
        finally:
            os.chdir(current_dir)
    
    def apply_patch(self, patch_string: str) -> bool:
        """
        Applies a given Git patch string to the repository.
        
        Args:
            patch_string (str): The Git patch string to apply.
        
        Returns:
            bool: True if the patch was applied successfully, False otherwise.
        """
        current_dir = os.getcwd()
        try:
            os.chdir(self.repo_path)
            
            # Apply the patch
            result = subprocess.run(
                ["git", "apply", "--whitespace=fix"], 
                input=patch_string, 
                capture_output=True, 
                text=True,
                check=True
            )
            logger.info(f"Patch applied successfully: {result.stdout}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error applying patch: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during patch application: {e}")
            return False
        finally:
            os.chdir(current_dir)

# Example Usage:
if __name__ == "__main__":
    dummy_repo_path = "/tmp/test_repo_for_patch"
    os.makedirs(dummy_repo_path, exist_ok=True)
    
    # Initialize a dummy git repo
    current_cwd = os.getcwd()
    os.chdir(dummy_repo_path)
    subprocess.run(["git", "init"], check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], check=True)
    
    with open("test_file.txt", "w") as f:
        f.write("Initial content\n")
    subprocess.run(["git", "add", "test_file.txt"], check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], check=True)
    
    os.chdir(current_cwd)
    
    patch_gen = PatchGenerator(repo_path=dummy_repo_path)
    
    # Make some changes
    with open(os.path.join(dummy_repo_path, "test_file.txt"), "a") as f:
        f.write("Added new line\n")
    
    # Generate patch
    generated_patch = patch_gen.generate_patch(commit_message="Added new line to test_file.txt")
    print(f"\nGenerated Patch:\n{generated_patch}")
    
    # Verify patch content (should contain the added line)
    assert "Added new line" in generated_patch
    
    # Clean up
    os.chdir(dummy_repo_path)
    subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=True)
    os.chdir(current_cwd)
    # shutil.rmtree(dummy_repo_path)