import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self,
                 kinematics_input_size,
                 image_input_channels=3,
                 actor_output_size=4,
                 input_image_size=(64, 48)):
        super().__init__()

        # CNN for image input
        target_estimator_output_size = 3
        self.target_estimator = nn.Sequential(
            nn.Conv2d(in_channels=image_input_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(32 * (input_image_size[0] // 4) * (input_image_size[1] // 4), 1024),
            nn.ReLU(),  # TODO: ajr -- experiment with these
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, target_estimator_output_size)
        )

        # Concatenate kinematics data with CNN output
        shared_output_size = 128
        self.shared_layer = nn.Sequential(
            nn.Linear(target_estimator_output_size + kinematics_input_size, 256),  # todo: ajr -- 256 or just 15?
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, shared_output_size),
        )

        # Two Linear layers with 'tanh' activation
        self.actor = nn.Linear(in_features=shared_output_size, out_features=actor_output_size)
        self.critic = nn.Linear(in_features=shared_output_size, out_features=1)

    def forward(self, image_input, kinematics_input):
        # Image input through CNN and fully connected head
        image_output = self.cnn(image_input)
        image_output = image_output.view(image_output.size(0), -1)
        image_output = self.fc_head(image_output)

        # Concatenate kinematics data with CNN output
        combined_data = torch.cat((image_output, kinematics_input), dim=1)
        combined_data = torch.tanh(self.concat_layer(combined_data))

        # Actor and critic outputs
        actor_output = torch.tanh(self.actor_layer(combined_data))
        critic_output = self.critic_layer(combined_data)

        # Final layer outputs
        final_outputs = self.final_layer(combined_data)

        return actor_output, critic_output, final_outputs