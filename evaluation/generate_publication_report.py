"""Compare actual held-out predictions using paired study coverage."""
if __package__:
    from .common import main
else:
    from common import main

if __name__ == "__main__":
    main(per_class=False)
