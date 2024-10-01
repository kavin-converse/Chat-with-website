class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.btnbot'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }

        this.state = false;
        this.messages = [];
    }

    display() {
        const {openButton, chatBox, sendButton} = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox))

        sendButton.addEventListener('click', () => this.onSendButton(chatBox))

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({key}) => {
            if (key === "Enter") {
                this.onSendButton(chatBox)
            }
        })
    }

    toggleState(chatbox) {
        this.state = !this.state;

        // show or hides the box
        if(this.state) {
            chatbox.classList.add('chatbox--active')
        } else {
            chatbox.classList.remove('chatbox--active')
        }

    function hideElement() {
            const element = document.querySelector('.botim');
            if (element) {
                element.style.display = 'none';
            }
        }
        
        // Event listener for the button click
        document.querySelector('.send__button').addEventListener('click', hideElement);
        
        // Event listener for the Enter key press
        document.addEventListener('keyup', function(event) {
            if (event.key === 'Enter') {
                hideElement();
            }
        });
        
    }
    

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value;
        if (text1 === "") {
            return; // Prevents sending empty messages
        }
        
        // Show user message immediately
        let msg1 = { name: "User", message: text1 };
        this.messages.push(msg1);
        this.updateChatText(chatbox); // Update chat with the user's message immediately
        textField.value = ''; // Clear input field right after sending the user message
        
        // Show loading message
        let loadingMsg = { name: "bot", message: 'Loading  <div class="wave-loader"><span></span> <span></span> <span></span> </div>' };
        this.messages.push(loadingMsg);
        this.updateChatText(chatbox); // Update chat to show loading message
    
        // Make the API call after showing the user message
        fetch('http://127.0.0.1:8000/ask', {
            method: 'POST',
            body: JSON.stringify({ question: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
        .then(r => r.json())
        .then(r => {
            console.log(r);
            // Remove loading message
            this.messages.pop(); // Remove the last (loading) message
            let msg2 = { name: "bot", message: r.response }; // Display response from API
            this.messages.push(msg2);
            this.updateChatText(chatbox); // Update chat with the API response
        }).catch((error) => {
            console.error('Error:', error);
            // Remove loading message
            this.messages.pop(); // Remove the last (loading) message
            let errorMsg = { name: "bot", message: "Sorry, something went wrong." };
            this.messages.push(errorMsg);
            this.updateChatText(chatbox); // Update chat with error message
        });
    }
    

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function(item) {
            if (item.name === "bot") {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>';
            } else if (item.message === "Loading...") {
                html += '<div class="messages__item messages__item--visitor loading-message">' + item.message + '</div>'; // Add specific class for loading message
            } else {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>';
            }
        });
    
        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }
    
}



const chatbox = new Chatbox();
chatbox.display();

function toggleChat() {
    const chatIcon = document.getElementById('chat-icon');
    const closeIcon = document.getElementById('close-icon');
    const chatboxSupport = document.querySelector('.chatbox__support');

    // Toggle chatbox visibility
    if (chatboxSupport.classList.contains('chatbox--active')) {
        chatboxSupport.classList.remove('chatbox--active');
        chatIcon.style.display = 'inline';  // Show the chat icon
        closeIcon.style.display = 'none';   // Hide the close (X) symbol
    } else {
        console.log("sdfwsfwe")
        chatboxSupport.classList.remove('hidden');
        chatboxSupport.classList.add('chatbox--active');
        chatIcon.style.display = 'none';    // Hide the chat icon
        closeIcon.style.display = 'inline'; // Show the close (X) symbol
    }
}

function closeTooltip() {
    const tooltip = document.getElementById('btnbot-tip');
    tooltip.style.display = 'none';
}