const std = @import("std");
const fiat_shamir = @import("fiat_shamir.zig");
const Transcript = fiat_shamir.Transcript;

test "Transcript initialization" {
    var transcript = Transcript.init("test_label");
    const initial_challenge = transcript.getChallenge("init_challenge");

    // Test determinism
    var transcript2 = Transcript.init("test_label");
    const initial_challenge2 = transcript2.getChallenge("init_challenge");

    try std.testing.expectEqualSlices(u8, &initial_challenge, &initial_challenge2);
}

test "Transcript absorb and squeeze" {
    var transcript = Transcript.init("test_label");
    transcript.appendMessage("msg1", "hello");
    transcript.appendU64("val1", 42);

    const challenge1 = transcript.getChallengeU64("chal1");

    var transcript2 = Transcript.init("test_label");
    transcript2.appendMessage("msg1", "hello");
    transcript2.appendU64("val1", 42);

    const challenge2 = transcript2.getChallengeU64("chal1");

    try std.testing.expectEqual(challenge1, challenge2);

    // Test binding - changing order or content changes challenge
    var transcript3 = Transcript.init("test_label");
    transcript3.appendU64("val1", 42);
    transcript3.appendMessage("msg1", "hello"); // Reversed order

    const challenge3 = transcript3.getChallengeU64("chal1");
    try std.testing.expect(challenge1 != challenge3);
}
