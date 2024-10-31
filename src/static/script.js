document.addEventListener("DOMContentLoaded", function() {
    const velocitySource = new EventSource("/velocity_feed");
    const statusSource = new EventSource("/status_feed");

    // Unified event handler for velocity feed
    velocitySource.onmessage = function(event) {
        const data = event.data;
        console.log("Received data:", data);

        // Adjust the regex pattern to match your exact data format
        const regex = /Average Speed:\s*([\d.]+)\s*mm\/s,\s*Predicted Speed:\s*([\d.]+)\s*mm\/s,\s*Predicted Current:\s*([\d.]+)A/;
        const matches = data.match(regex);

        if (matches) {
            const averageVelocity = matches[1];
            const predictedVelocity = matches[2];
            const predictedCurrent = matches[3];

            document.getElementById("velocity-value").textContent = averageVelocity + " mm/s";
            document.getElementById("predicted-velocity-value").textContent = predictedVelocity + " mm/s";
            document.getElementById("predicted-current-value").textContent = predictedCurrent + "A";
        } else {
            console.error("Unable to parse server-sent event data:", data);
        }
    };

    // Status event handler
    statusSource.onmessage = function(event) {
        const status = event.data;
        const statusTextElement = document.getElementById("statusText");

        if (status === "NG") {
            statusTextElement.textContent = "NG";
            statusTextElement.parentElement.classList.replace("status-ok", "status-ng");
        } else {
            statusTextElement.textContent = "OK";
            statusTextElement.parentElement.classList.replace("status-ng", "status-ok");
        }
    };

    // Event listener for the "Submit Current" button
    document.getElementById('submit-current').addEventListener('click', function(e) {
        e.preventDefault();
        var inputCurrentValue = document.getElementById('input-current-value').value;
        if (!inputCurrentValue || inputCurrentValue == "0") {
            alert("Please input welding data.");
            return;
        }

        $.ajax({
            url: '/submit_current',
            type: 'POST',
            data: {'currentValue': inputCurrentValue},
            success: function(response) {
                if (response.status === "crack") {
                    document.getElementById('predicted-current-value').textContent = response.current + "A";
                    console.log('Success:', response);
                } else {
                    document.getElementById('predicted-current-value').textContent = inputCurrentValue + "A";
                    document.getElementById('predicted-velocity-value').textContent = document.getElementById("velocity-value").textContent;
                }
            },
            error: function(error) {
                console.log('Error:', error);
            }
        });
    });
});
