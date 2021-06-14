$(document).ready(function () {
    $('#tabs li').on('click', function () {
        console.log('Heyy')
        const tab = $(this).data('tab');

        $('#tabs li').removeClass('is-active');
        $(this).addClass('is-active');

        $('#tab-content div').removeClass('is-active');
        $('div[data-content="' + tab + '"]').addClass('is-active');
    });
});