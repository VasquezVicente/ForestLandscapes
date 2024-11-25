from detect import detect_crowns

def main(config_path):
    # Detect objects using the configuration file
    detect_crowns(config_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    main(config_path)