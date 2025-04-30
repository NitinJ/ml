import re

def clean_chat(input_file='data/_chat.txt', output_file='data/output_chat.txt'):
    pattern = re.compile(r'^\[\d{1,2}/\d{1,2}/\d{2,4}, .*?\] (.*?): (.*)')
    ENCRYPTION_NOTICE = "Messages and calls are end-to-end encrypted. Only people in this chat can read, listen to, or share them"

    output_lines = []
    last_sender = None
    message_accumulator = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ENCRYPTION_NOTICE in line:
                continue

            match = pattern.match(line)
            if match:
                sender, message = match.groups()
                if sender == last_sender:
                    message_accumulator.append(message)
                else:
                    if last_sender is not None:
                        combined = '. '.join(message_accumulator)
                        output_lines.append(f"{last_sender}: {combined}")
                    last_sender = sender
                    message_accumulator = [message]
            else:
                if message_accumulator:
                    message_accumulator.append(line)

    if last_sender and message_accumulator:
        combined = '. '.join(message_accumulator)
        output_lines.append(f"{last_sender}: {combined}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')

def clean_other_stuff():
    import re

    with open('data/output_chat.txt', 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace timestamps of the form "[MM/DD/YY, HH:MM:SS AM/PM]" (with optional invisible char) with newline
    content = re.sub(r'\u200e?\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s[AP]M\]', '\n', content)

    with open('data/output_chat.txt', 'w', encoding='utf-8') as f:
        f.write(content)

def cleanup():
    with open('data\output_chat.txt', 'r', encoding='utf-8') as f:
        content = f.read()

        # This regex finds parts of lines that contain \u200e and removes text until the next '.'
        # Explanation:
        # - [^.]*?\u200e.*?\.: match the smallest segment from start until `\u200e`, consume till the next dot (non-greedy)
        # - Flags=re.DOTALL to handle multi-line cases safely
        cleaned_content = re.sub(r'[^.]*?\u200e.*?\.', '', content)

        # Optional: strip extra whitespace
        cleaned_content = re.sub(r'\s{2,}', ' ', cleaned_content).strip()

        with open('data\output_chat.txt', 'w', encoding='utf-8') as f:
            f.write(cleaned_content)


if __name__ == "__main__":
    clean_chat()
    clean_other_stuff()
    cleanup()
