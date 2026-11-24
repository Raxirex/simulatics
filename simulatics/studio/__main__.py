from .app import main

if __name__ == "__main__":      # worker processes (design search) re-import this module
    raise SystemExit(main())
