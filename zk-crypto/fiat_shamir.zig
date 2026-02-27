const std = @import("std");

/// A cryptographic transcript for the Fiat-Shamir transform using Blake3
pub const Transcript = struct {
    hasher: std.crypto.hash.Blake3,

    pub fn init(label: []const u8) Transcript {
        var transcript = Transcript{
            .hasher = std.crypto.hash.Blake3.init(.{}),
        };
        transcript.appendMessage("init", label);
        return transcript;
    }

    /// Absorb data into the transcript
    pub fn appendMessage(self: *Transcript, label: []const u8, message: []const u8) void {
        self.hasher.update(label);
        self.hasher.update(message);
    }

    /// Absorb a u64 value into the transcript
    pub fn appendU64(self: *Transcript, label: []const u8, value: u64) void {
        var buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &buf, value, .little);
        self.appendMessage(label, &buf);
    }

    /// Squeeze a cryptographic challenge from the transcript
    pub fn getChallenge(self: *Transcript, label: []const u8) [32]u8 {
        self.hasher.update(label);

        var challenge: [32]u8 = undefined;
        // We clone here so that we don't finalize the internal hasher
        // This allows continued appending to the main transcript.
        var hasher_copy = self.hasher;
        hasher_copy.final(&challenge);

        // Feed the challenge back into the main hasher so future state depends on it
        self.hasher.update(&challenge);
        return challenge;
    }

    /// Extract a u64 challenge (simulating a scalar in a smaller field)
    pub fn getChallengeU64(self: *Transcript, label: []const u8) u64 {
        const full_challenge = self.getChallenge(label);
        return std.mem.readInt(u64, full_challenge[0..8], .little);
    }
};
